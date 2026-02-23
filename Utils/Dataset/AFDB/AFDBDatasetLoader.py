search_paths = [
            '/content/drive/MyDrive/AFDB/physionet.org/files/afdb/1.0.0',
            '/content/mit-bih-atrial-fibrillation-database-1.0.0',
            '/content/afdb',
            'mit-bih-atrial-fibrillation-database-1.0.0',
            'afdb',
            './afdb',
            '.',
        ]
from typing import Optional
import wfdb

from Utils.Dataset.DatasetLoader import DatasetLoader

class AFDBDatasetLoader(DatasetLoader):
    def __init__(self, 
                 data_path: Optional[str] = None, 
                 use_physionet: bool = False):
        super().__init__(
            dataset_name='afdb',
            search_paths=search_paths,
            data_path=data_path,
            use_physionet=use_physionet
        )
    def parse_rhythm_segments(self, annotation: wfdb.Annotation, fs: int, total_samples: int):
        """Parse rhythm segments (AF and SR only)"""
        segments = []

        rhythm_indices = [i for i, (sym, aux) in enumerate(zip(annotation.symbol, annotation.aux_note))
                         if sym.startswith('(') or (aux and aux.startswith('('))]

        for i, idx in enumerate(rhythm_indices):
            rhythm = annotation.symbol[idx]
            if rhythm == '+' and annotation.aux_note[idx]:
                rhythm = annotation.aux_note[idx]

            rhythm = rhythm.strip()
            start_sample = annotation.sample[idx]

            if i < len(rhythm_indices) - 1:
                end_sample = annotation.sample[rhythm_indices[i + 1]]
            else:
                end_sample = total_samples

            # Only keep AF and SR (exclude AFL, AV junction)
            if rhythm == '(AFIB':
                rhythm_label = 'AF'
            elif rhythm == '(N':
                rhythm_label = 'SR'
            else:
                continue  # Skip other rhythms

            start_time = start_sample / fs
            end_time = end_sample / fs
            duration = end_time - start_time

            # For AF, only keep segments >= 30 seconds
            if rhythm_label == 'AF' and duration < 30:
                continue

            segments.append({
                'rhythm': rhythm_label,
                'start_time': start_time,
                'end_time': end_time,
                'duration': duration,
                'start_sample': start_sample,
                'end_sample': end_sample
            })

        return segments