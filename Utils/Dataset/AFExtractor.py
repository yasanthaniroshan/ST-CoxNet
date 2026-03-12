from typing import Optional, List, Dict
import os
import numpy as np
import pandas as pd
import wfdb

from Utils.Dataset.AFEpisode import AFEpisode


class AFExtractor:
    """Extract and analyze AF episodes and export in WFDB format"""

    def __init__(self, loader, AF_low, preAF_low, preAF_max, output_dir: Optional[str] = None):
        self.loader = loader
        self.AF_low = AF_low
        self.preAF_low = preAF_low
        self.preAF_max = preAF_max
        self.OUTPUT_DIR = output_dir if output_dir else 'extracted_segments'
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)

        self.extraction_report = []  # Detailed extraction report
        self.results = {}
        self.total_extracted = 0

    def extract_af_episodes(self, segments: List[Dict], fs: int, total_samples: int) -> List[AFEpisode]:
        """Extract AF episodes with analysis"""
        episodes = []
        af_segments = [s for s in segments if s['rhythm'] == 'AF']

        for ep_num, af_seg in enumerate(af_segments, 1):
            # Find pre rhythm
            pre_rhythm = None
            available_sr_before = 0.0

            # Check what's before this AF episode
            pre_segments = [s for s in segments if s['end_sample'] <= af_seg['start_sample']]

            if pre_segments:
                pre_rhythm = pre_segments[-1]['rhythm']

                # Calculate continuous SR duration before AF
                if pre_rhythm == 'SR':
                    sr_duration = 0.0
                    for seg in reversed(pre_segments):
                        if seg['rhythm'] == 'SR':
                            sr_duration += seg['duration']
                        else:
                            break
                    available_sr_before = sr_duration / 60.0  # Convert to minutes

            # Create AFEpisode object
            episodes.append(AFEpisode(
                episode_number=ep_num,
                start_sample=af_seg['start_sample'],
                end_sample=af_seg['end_sample'],
                start_time=af_seg['start_time'],
                end_time=af_seg['end_time'],
                duration=af_seg['duration'],
                duration_minutes=af_seg['duration'] / 60.0,
                pre_rhythm=pre_rhythm,
                available_sr_before=available_sr_before,
                starts_recording=(af_seg['start_sample'] == 0),
                ends_recording=(af_seg['end_sample'] >= total_samples - fs * 60)
            ))

        return episodes

    def save_wfdb_record(self, ecg_data: np.ndarray, fs: int, record_name: str,
                         original_record: wfdb.Record):
        """Save ECG segment in WFDB format (.hea, .dat)"""
        output_path = os.path.join(self.OUTPUT_DIR, record_name)

        # Prepare signal names and units from original record
        sig_name = original_record.sig_name if hasattr(original_record, 'sig_name') else ['ECG1', 'ECG2']
        units = original_record.units if hasattr(original_record, 'units') else ['mV', 'mV']

        # Handle single or dual channel
        if ecg_data.ndim == 1:
            ecg_data = ecg_data.reshape(-1, 1)
            sig_name = [sig_name[0]]
            units = [units[0]]
        elif ecg_data.shape[1] == 1:
            sig_name = [sig_name[0]]
            units = [units[0]]

        # Write the record
        wfdb.wrsamp(
            record_name=record_name,
            fs=fs,
            units=units[:ecg_data.shape[1]],
            sig_name=sig_name[:ecg_data.shape[1]],
            p_signal=ecg_data,
            write_dir=self.OUTPUT_DIR
        )

    def save_qrs_annotation(self, qrs_annotation: wfdb.Annotation,
                           segment_start_sample: int, segment_end_sample: int,
                           record_name: str, fs: int):
        """Extract and save QRS annotations for the segment"""
        if qrs_annotation is None:
            return

        # Find QRS beats within segment
        qrs_samples = qrs_annotation.sample
        qrs_symbols = qrs_annotation.symbol

        # Filter QRS beats within segment
        mask = (qrs_samples >= segment_start_sample) & (qrs_samples < segment_end_sample)
        segment_qrs_samples = qrs_samples[mask]
        segment_qrs_symbols = [qrs_symbols[i] for i in range(len(qrs_symbols)) if mask[i]]

        if len(segment_qrs_samples) == 0:
            return

        # Adjust sample indices to be relative to segment start
        relative_samples = segment_qrs_samples - segment_start_sample

        # Write QRS annotation
        output_path = os.path.join(self.OUTPUT_DIR, record_name)

        qrs_ann = wfdb.Annotation(
            record_name=record_name,
            extension='qrs',
            sample=relative_samples,
            symbol=segment_qrs_symbols
        )
        qrs_ann.wrann(write_dir=self.OUTPUT_DIR)

    def save_rhythm_annotation(self, af_start_sample: int, af_end_sample: int,
                               total_samples: int, record_name: str):
        """Create and save rhythm annotation (.atr) marking SR and AF regions"""

        # Create annotation samples and symbols
        ann_samples = [0, af_start_sample, af_end_sample]
        ann_symbols = ['+', '+', '+']
        ann_aux = ['(N', '(AFIB', '(N']  # SR before AF, AF episode, SR after (if any)

        # If there's no data after AF, remove the last annotation
        if af_end_sample >= total_samples:
            ann_samples = ann_samples[:-1]
            ann_symbols = ann_symbols[:-1]
            ann_aux = ann_aux[:-1]

        # Write rhythm annotation
        rhythm_ann = wfdb.Annotation(
            record_name=record_name,
            extension='atr',
            sample=np.array(ann_samples),
            symbol=ann_symbols,
            aux_note=ann_aux
        )
        rhythm_ann.wrann(write_dir=self.OUTPUT_DIR)

    def count_and_extract_possible_segments_per_record(self, record_id: str) -> Dict:
        """Analyze and extract AF episodes with full WFDB format"""

        print(f"Analyzing {record_id}... ", end="")

        record, annotation = self.loader.load_record(record_id)
        qrs_annotation = self.loader.load_qrs_annotation(record_id)

        if record is None or annotation is None:
            print("Failed to load")
            return {
                'record_id': record_id,
                'status': 'failed',
                'total_af_episodes': 0,
                f'af_episodes_{self.AF_low}s': 0,
                f'extractable_{self.preAF_low}_{self.preAF_max}': 0,
                'extracted_count': 0
            }

        fs = int(record.fs)
        total_samples = record.sig_len

        # Parse rhythm segments
        segments = self.loader.parse_rhythm_segments(annotation, fs, total_samples)

        if not segments:
            print("No AF/SR segments")
            return {
                'record_id': record_id,
                'status': 'no_segments',
                'total_af_episodes': 0,
                f'af_episodes_{self.AF_low}s': 0,
                f'extractable_{self.preAF_low}_{self.preAF_max}': 0,
                'extracted_count': 0
            }

        # Extract AF episodes
        episodes = self.extract_af_episodes(segments, fs, total_samples)

        # Count episodes meeting criteria
        total_af = len(episodes)
        extractable_eps = sum(1 for ep in episodes if ep.duration >= self.AF_low)

        # Filter episodes for extraction: AF >= AF_low AND pre-AF SR >= preAF_low min AND not at recording start
        extractable_episodes = [
            ep for ep in episodes
            if ep.duration >= self.AF_low 
            and self.preAF_low <= ep.available_sr_before 
            and not ep.starts_recording
        ]

        extract_count = len(extractable_episodes)

        # Extract and save ECG segments with full WFDB format
        for episode in extractable_episodes:
            # Determine pre-AF window (limit to preAF_max minutes)
            pre_af_minutes = min(self.preAF_max, episode.available_sr_before)
            pre_af_samples = int(pre_af_minutes * 60 * fs)

            # Calculate segment boundaries
            segment_start_sample = episode.start_sample - pre_af_samples
            segment_end_sample = episode.end_sample

            # Extract ECG segment (both channels if available)
            ecg_segment = record.p_signal[segment_start_sample:segment_end_sample, :]

            # Generate record name
            segment_name = f"{record_id}_ep{episode.episode_number:02d}"

            # 1. Save ECG data in WFDB format (.hea and .dat)
            self.save_wfdb_record(ecg_segment, fs, segment_name, record)

            # 2. Save QRS annotations for the segment
            self.save_qrs_annotation(qrs_annotation, segment_start_sample,
                                    segment_end_sample, segment_name, fs)

            # 3. Save rhythm annotation (.atr) marking SR and AF
            af_start_in_segment = pre_af_samples
            af_end_in_segment = pre_af_samples + int(episode.duration * fs)
            self.save_rhythm_annotation(af_start_in_segment, af_end_in_segment,
                                       len(ecg_segment), segment_name)

            # 4. Add to extraction report
            self.extraction_report.append({
                'Record_ID': record_id,
                'Episode_Number': episode.episode_number,
                'Segment_Name': segment_name,
                'Pre_AF_Minutes_Extracted': pre_af_minutes,
                'AF_Duration_Minutes': episode.duration_minutes,
                'Total_Duration_Minutes': pre_af_minutes + episode.duration_minutes,
                'AF_Start_Time': episode.start_time / 60.0,  # in minutes
                'AF_End_Time': episode.end_time / 60.0,  # in minutes
                'Available_SR_Before': episode.available_sr_before,
                'Sampling_Frequency': fs
            })

            self.total_extracted += 1

        print(f"✓ AF: {total_af}, AF≥{self.AF_low}s: {extractable_eps}, Extractable: {extract_count}, Extracted: {extract_count}")

        result = {
            'record_id': record_id,
            'status': 'success',
            'total_af_episodes': total_af,
            f'af_episodes_{self.AF_low}s': extractable_eps,
            f'extractable_{self.preAF_low}_{self.preAF_max}': extract_count,
            'extracted_count': extract_count,
            'episodes': episodes
        }

        self.results[record_id] = result
        return result

    def analyze_all_records(self) -> pd.DataFrame:
        """Analyze all records and generate comprehensive reports"""
        print(f"\n{'='*80}")
        print("AF EPISODE EXTRACTION AND EXPORT (WFDB FORMAT)")
        print(f"{'='*80}\n")
        print(f"Criteria:")
        print(f"  • AF duration ≥ {self.AF_low} seconds")
        print(f"  • Pre-AF SR window ≥ {self.preAF_low} minutes (max {self.preAF_max} minutes)")
        print(f"  • Episode does not start at recording beginning")
        print(f"\nOutput directory: {self.OUTPUT_DIR}")
        print(f"Output format: WFDB (.hea, .dat, .qrs, .atr)")
        print(f"{'='*80}\n")

        if not self.loader.record_ids:
            print("ERROR: No records found!")
            return pd.DataFrame()

        print(f"Total records to analyze: {len(self.loader.record_ids)}\n")

        all_results = []

        for record_id in self.loader.record_ids:
            result = self.count_and_extract_possible_segments_per_record(record_id)
            summary = {k: v for k, v in result.items() if k != 'episodes'}
            all_results.append(summary)

        # Create summary DataFrame
        df = pd.DataFrame(all_results)

        # Save detailed extraction report
        if self.extraction_report:
            report_path = os.path.join(self.OUTPUT_DIR, 'EXTRACTION_REPORT.csv')
            df_report = pd.DataFrame(self.extraction_report)
            df_report.to_csv(report_path, index=False)

            print(f"\n{'='*80}")
            print(f"✓ Saved EXTRACTION_REPORT.csv with {len(df_report)} segments")
            print(f"  Location: {report_path}")
            print(f"\nExtraction Report Preview:")
            print(df_report.to_string(index=False, max_rows=10))

        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")

        if len(df) > 0:
            successful = df[df['status'] == 'success']

            print(f"\nSuccessfully analyzed: {len(successful)}/{len(df)} records")
            print(f"\nTotal Statistics:")
            print(f"  • Total AF episodes: {successful['total_af_episodes'].sum()}")
            print(f"  • AF episodes ≥{self.AF_low}s: {successful[f'af_episodes_{self.AF_low}s'].sum()}")
            print(f"  • Extractable episodes (≥{self.preAF_low}min pre-AF SR): {successful[f'extractable_{self.preAF_low}_{self.preAF_max}'].sum()}")
            print(f"  • Total segments extracted: {self.total_extracted}")

            # Records with most extractable episodes
            print(f"\nTop 5 Records by Extractable Episodes:")
            extractable_col = f'extractable_{self.preAF_low}_{self.preAF_max}'
            top5 = successful.nlargest(5, extractable_col)[
                ['record_id', extractable_col, 'extracted_count']
            ]
            for idx, row in top5.iterrows():
                print(f"  {row['record_id']}: {row[extractable_col]} extractable, "
                      f"{row['extracted_count']} extracted")

        # print(f"\n{'='*80}")
        # print("OUTPUT FILES")
        # print(f"{'='*80}")
        # print(f"For each extracted segment (e.g., 04015_ep01):")
        # print(f"  • 04015_ep01.hea  - WFDB header file")
        # print(f"  • 04015_ep01.dat  - ECG signal data (both leads)")
        # print(f"  • 04015_ep01.qrs  - QRS annotations")
        # print(f"  • 04015_ep01.atr  - Rhythm annotations (SR/AF)")
        # print(f"\nAll files saved to: {self.OUTPUT_DIR}/")
        # print(f"Extraction report: {os.path.join(self.OUTPUT_DIR, 'EXTRACTION_REPORT.csv')}")

        # print(f"\n{'='*80}")
        # print("USAGE IN PYTHON")
        # print(f"{'='*80}")
        # print(f"# Read a segment")
        # print(f"record = wfdb.rdrecord('{self.OUTPUT_DIR}/04015_ep01')")
        # print(f"qrs = wfdb.rdann('{self.OUTPUT_DIR}/04015_ep01', 'qrs')")
        # print(f"rhythm = wfdb.rdann('{self.OUTPUT_DIR}/04015_ep01', 'atr')")
        # print(f"")
        # print(f"# Access data")
        # print(f"ecg_data = record.p_signal  # ECG signals (both leads)")
        # print(f"fs = record.fs  # Sampling frequency")
        # print(f"qrs_samples = qrs.sample  # QRS beat locations")
        # print(f"rhythm_segments = rhythm.aux_note  # ['(N', '(AFIB', ...]")
        # print(f"{'='*80}\n")

        return df