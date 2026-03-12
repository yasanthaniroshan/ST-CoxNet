from collections import Counter
import sys
import os
from typing import Dict, Any
from sklearn.linear_model import LinearRegression
import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../..'))

import neurokit2 as nk
import numpy as np
from Utils.FeatureExtractor.Base import BaseExtractor
from neurokit2.hrv.hrv_utils import _hrv_format_input
import warnings

# Suppress all NeuroKit2 warnings more comprehensively
warnings.filterwarnings("ignore", category=UserWarning, module="neurokit2")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="neurokit2")
warnings.filterwarnings("ignore", message=".*DFA_alpha2.*")
warnings.filterwarnings("ignore", message=".*long-term correlation.*")
warnings.filterwarnings("ignore", message=".*invalid value encountered in scalar divide.*")
warnings.filterwarnings("ignore", message=".*entropy_multiscale.*")
# Also try catching it as a general warning
warnings.simplefilter("ignore", category=UserWarning)
warnings.simplefilter("ignore", category=RuntimeWarning)

def _filter(d: Dict[str, Any], flags: Dict[str, int]) -> Dict[str, Any]:
    """Keep only keys present in flags with value == 1."""
    return {k: v for k, v in d.items() if flags.get(k, 0) == 1}
    

class HRVFeatures(BaseExtractor):
    def __init__(self, data, fs, rri_given=False, config_path="Utils/FeatureExtractor/config.yaml"):
        self.data = data
        self.fs = fs
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        self._flags: Dict[str, Dict[str, int]] = cfg.get("features", {})
        rqa_params = cfg.get("rqa_params", {})
        self.m        = rqa_params.get("m", 3)
        self.tau      = rqa_params.get("tau", 1)
        self.rr_pct   = rqa_params.get("rr_percent", 0.03)
        self.l_min    = rqa_params.get("l_min", 2)
        self.v_min    = rqa_params.get("v_min", 2)

        if rri_given:
            self.rri = data # Assuming data is already in miliseconds.
            self.peaks = nk.intervals_to_peaks(self.rri, sampling_rate=self.fs)
        else:
            self.peaks, _ = nk.ecg_peaks(data, sampling_rate=fs)
            self.rri, _, _ = _hrv_format_input(self.peaks, sampling_rate=fs)
    
    def compute_time_features(self):
        flags = self._flags.get("time_domain", {})
        if not any(flags.values()):
            return {}
        hrv_time = nk.hrv_time(self.peaks, sampling_rate=self.fs, show=False)
        features = hrv_time.to_dict(orient='records')[0] if not hrv_time.empty else {}
        features["NN50"] = self.NN50()
        features["pNN31"] = self.pNN31()
        features.update(self.HR())
        return _filter(features, flags)

    def compute_frequency_features(self):
        flags = self._flags.get("frequency_domain", {})
        if not any(flags.values()):
            return {}
        hrv_frequency = nk.hrv_frequency(self.peaks, sampling_rate=self.fs, show=False, normalize=True)
        features = hrv_frequency.to_dict(orient='records')[0] if not hrv_frequency.empty else {}
        return _filter(features, flags)
    
    def compute_nonlinear_features(self):
        flags = self._flags.get("nonlinear", {})
        if not any(flags.values()):
            return {}
        hrv_nonlinear = nk.hrv_nonlinear(self.peaks, sampling_rate=self.fs, show=False)
        features = hrv_nonlinear.to_dict(orient='records')[0] if not hrv_nonlinear.empty else {}
        features["RENYIEN"] = self.RENYIEN(alpha=2)
        features["TSALLISEN"] = self.TSALLISEN(q=2)
        return _filter(features, flags)
    
    def compute_rqa_features_fixed_rr(self,exclude_loi=True):
        flags = self._flags.get("rqa", {})
        if not any(flags.values()):
            return {}
        rqa_features = self.RQA(self.m, self.tau, self.rr_pct, self.l_min, self.v_min, exclude_loi=exclude_loi)
        return _filter(rqa_features, flags)
    
    def compute_rqa(self):
        flags = self._flags.get("rqa_nk", {})
        if not any(flags.values()):
            return {}
        rqa_nk = nk.hrv_rqa(self.peaks, sampling_rate=self.fs, dimension=self.m, delay=self.tau, show=False)
        features = rqa_nk.to_dict(orient='records')[0] if not rqa_nk.empty else {}
        return _filter(features, flags)
    
    def compute_all(self, exclude_loi=True):
        features = {}
        features.update(self.compute_time_features())
        features.update(self.compute_frequency_features())
        features.update(self.compute_nonlinear_features())
        features.update(self.compute_rqa_features_fixed_rr(exclude_loi=exclude_loi))
        features.update(self.compute_rqa())
        return features
    
    def NN50(self):
        diff_rri = np.diff(self.rri)
        nn50_count = np.sum(np.abs(diff_rri) > 50)
        return nn50_count
    
    def pNN31(self):
        diff_rri = np.diff(self.rri)
        nn31_count = np.sum(np.abs(diff_rri) > 31)
        return nn31_count / (len(diff_rri) + 1) * 100
    
    def HR(self):
        hr = 60000 / self.rri
        return {
            "Mean_HR": np.mean(hr), 
            "SD_HR": np.std(hr, ddof=1), 
            "Min_HR": np.min(hr),
            "Max_HR": np.max(hr)
        }

    def __rri_to_prob(self, bins="fd"):
        hist, bin_edges = np.histogram(self.rri, bins=bins, density=True)
        p = hist * np.diff(bin_edges)   
        p = p[p > 0]                    
        return p
    
    def RENYIEN(self, alpha=2):
        p = self.__rri_to_prob()
        if len(p) == 0:
            return 0.0
        if alpha == 1:
            return -np.sum(p * np.log(p + 1e-12)) 
        else:
            return (1 / (1 - alpha)) * np.log(np.sum(p ** alpha))
    
    def TSALLISEN(self, q=2):
        p = self.__rri_to_prob()
        if len(p) == 0:
            return 0.0
        if q == 1:
            return -np.sum(p * np.log(p + 1e-12)) 
        else:
            return (1 / (q - 1)) * (1 - np.sum(p ** q))

    def RQA(self, m , tau, rr_percent, l_min, v_min, exclude_loi=True):
        rr_intervals = np.asarray(self.rri)
        N = len(rr_intervals)
        M = N - (m - 1)*tau
        if N <= 0:
            raise ValueError("Time series too short for given m and tau")
        state_space =  np.zeros((M, m))

        for i in range(M):
            state_space[i] = rr_intervals[i: i + m * tau : tau]
        
        C = np.cov(state_space, rowvar=False)
        eigvals = np.linalg.eigvalsh(C)
        eigvals = np.sort(eigvals)[::-1]

        norm = np.sum(eigvals)
        p = eigvals / (norm + 1e-12)

        NME = eigvals[0] / (norm + 1e-12)
        EEV = -np.sum(p * np.log(p + 1e-12))    

        D = np.max(np.abs(state_space[:, None, :] - state_space[None, :, :]), axis=2)
        len_D = D.shape[0]
        upper_triangle = D[np.triu_indices(len_D, k=1)]
        eps = np.percentile(upper_triangle, rr_percent * 100)
        # print(f"Calculated eps for RQA: {eps}")

        R = (D <= eps).astype(int)
        len_R = R.shape[0]
        Rb = (np.asarray(R) != 0).astype(np.uint8)

        if exclude_loi:
            np.fill_diagonal(Rb, 0)

        # Recurrence Rate
        recurrence_rate = np.sum(Rb) / (len_R ** 2) if len_R > 0 else 0.0

        counts = self._p_of_l(Rb, len_R)
        counts_lmin = {l: c for l, c in counts.items() if l >= l_min}

        counts_vertical = self._p_of_v(Rb, M)
        counts_vertical_lmin = {v: c for v, c in counts_vertical.items() if v >= v_min}

        total_counts = sum(counts_lmin.values())
        sum_l_pl = sum(L * c for L, c in counts_lmin.items())
        sum_l_pl_all = sum(L * c for L, c in counts.items())

        # Determinism
        determinism = sum_l_pl / sum_l_pl_all if sum_l_pl_all > 0 else 0.0

        # L mean
        L_mean = sum_l_pl / total_counts if total_counts > 0 else 0.0

        # L max
        L_max = max(counts_lmin.keys()) if counts_lmin else 0.0

        # Median diagonal line length
        if counts_lmin:
            lengths_expanded = []
            for L, c in counts_lmin.items():
                lengths_expanded.extend([L] * c)
            L_median = np.median(lengths_expanded)
        else:
            L_median = np.nan

        # Entropy
        probs = {L: c / total_counts for L, c in counts_lmin.items()} if total_counts > 0 else {}
        entropy = float(-sum(p * np.log(p) for p in probs.values()) if probs else 0.0)

        # Divergence - Can use as Lyapunov exponent
        divergence = 1.0 / L_max if L_max > 0 else 0.0

        ratio = determinism / recurrence_rate if recurrence_rate > 0 else 0.0

        counts_vertical_sum = sum(counts_vertical_lmin.values())
        sum_v_pl = sum(v * c for v, c in counts_vertical_lmin.items())
        sum_v_pl_all = sum(v * c for v, c in counts_vertical.items())

        # Laminarity (LAM)
        laminarity = sum_v_pl / sum_v_pl_all if sum_v_pl_all > 0 else 0.0

        # Trapping time (TT)
        trapping_time = sum_v_pl / counts_vertical_sum if counts_vertical_sum > 0 else 0.0

        # Maximum vertical line length (V_max)
        V_max = max(counts_vertical_lmin.keys()) if counts_vertical_lmin else 0.0

        # Vertical line entropy (V_entr)
        probs_v = {v: c / counts_vertical_sum for v, c in counts_vertical_lmin.items()} if counts_vertical_sum > 0 else {}
        V_entr = float(-sum(p * np.log(p) for p in probs_v.values()) if probs_v else 0.0)

        # 9. White vertical lines (time between recurrences)
        white_vertical_lengths = []
        for j in range(M):
            col = Rb[:, j]
            # Find runs of zeros (non-recurrent points)
            zero_runs = self._lengths_of_consecutive_ones(1 - col)  # Invert: 0->1, 1->0
            white_vertical_lengths.extend(zero_runs)

        W_mean = np.mean(white_vertical_lengths) if white_vertical_lengths else np.nan
        W_max = np.max(white_vertical_lengths) if white_vertical_lengths else np.nan

        pc = self._compute_pc_l(Rb, L_max)
        lg_pc = self._log_pc(pc)
        l_vals = np.arange(1, L_max + 1)
        K2 = self.K2(l_vals, lg_pc)
        features = {
            # Eigenvalue-based features
            'NME' : NME,
            'EEV' : EEV,
            
            # Diagonal Features
            'RecurrenceRate_RRFixed' : recurrence_rate,
            'Determinism_RRFixed' : determinism,
            'L_RRFixed' : L_mean,
            'LMax_RRFixed' : L_max,
            'LMedian_RRFixed' : L_median,
            'LEn_RRFixed' : entropy,
            'Divergence_RRFixed' : divergence,
            'Ratio_RRFixed' : ratio,
            'LMinCount_RRFixed' : total_counts,

            # Vertical Features
            'Laminarity_RRFixed' : laminarity,
            'TrappingTime_RRFixed' : trapping_time,
            'VMaX_RRFixed' : V_max,
            'VEn_RRFixed' : V_entr,

            # White Vertical Features
            'W_RRFixed' : W_mean,
            'WMax_RRFixed' : W_max,

            # Advanved features
            'K2' : K2
        }

        return features

    def _lengths_of_consecutive_ones(self, arr):
        array = np.asarray(arr, dtype=np.int8)
        if array.size == 0:
            return np.array([], dtype=int)
        edges = np.diff(np.r_[0, array, 0])
        starts = np.flatnonzero(edges == 1)
        ends = np.flatnonzero(edges == -1)
        return (ends - starts).tolist()

    def _p_of_l(self, Rb: np.ndarray, M: int):
        counts =        Counter()
        for k in range(-(M - 1), M):
            diag = np.diagonal(Rb, offset=k)
            for L in self._lengths_of_consecutive_ones(diag):
                counts[L] += 1
        return dict(counts)

    def _p_of_v(self, Rb: np.ndarray, M: int):
        counts = {}
        for j in range(M):
            col = Rb[:, j]
            runs = self._lengths_of_consecutive_ones(col)
            for v in runs:
                counts[v] = counts.get(v, 0) + 1
        return counts

    def _compute_pc_l(self, Rb: np.ndarray, l_max: int):
        N = Rb.shape[0]
        diag_lengths = []

        for k in range(-N + 1, N):
            diag = np.diagonal(Rb, offset=k)
            # print("Diagonal now:", diag)
            count = 0
            for val in diag:
                if val == 1:
                    count += 1
                else:
                    if count > 0:
                        diag_lengths.append(count)
                    count = 0
            if count > 0:
                diag_lengths.append(count)

        diag_lengths = np.array(diag_lengths)
        # print("Diagonal Lengths:", diag_lengths)
        pc = []
        for l in range(1, l_max + 1):
            pc.append(np.sum(diag_lengths >= l))
        pc = np.array(pc, dtype=float)
        pc /= pc[0]  # normalize by the diagonal
        return pc
    
    def _log_pc(self, pc: np.ndarray):
        pc = np.clip(pc, 1e-12, None)
        return np.log(pc)

    def K2(
        self, 
        l_vals: np.ndarray ,
        log_pc_vals: np.ndarray,
        min_len=3,
        r2_threshold=0.95,
        min_slope = -0.001) -> float:
        best_score = -np.inf
        K2 = None

        for i in range(len(l_vals)):
            if l_vals[i] < 3:
                continue
            for j in range(i + min_len, len(l_vals) + 1):
                x = l_vals[i:j].reshape(-1, 1)
                y = log_pc_vals[i:j]
                model = LinearRegression().fit(x, y)
                r2 = model.score(x, y)
                slope = model.coef_[0]
                if r2 >= r2_threshold and slope <= min_slope:
                    length = j - i
                    mean_l = np.mean(l_vals[i:j])
                    score = length * np.log(mean_l + 1)

                    if score > best_score:
                        best_score = score
                        K2 = -slope

            return K2

    

    def __helper__(self):
        """
        Provides a quick summary and interpretation of HRV time-domain features

        References:
        [1] Yin Z, Liu C, Xie C, Nie Z, Wei J, Zhang W and Liang H (2025) Identification of atrial fibrillation using heart rate variability: a meta-analysis. Front. Cardiovasc. Med. 12:1581683. doi: 10.3389/fcvm.2025.1581683
        [2] H. Costin, C. Rotariu and A. Păsărică, "Atrial fibrillation onset prediction using variability of ECG signals," 2013 8TH INTERNATIONAL SYMPOSIUM ON ADVANCED TOPICS IN ELECTRICAL ENGINEERING (ATEE), Bucharest, Romania, 2013, pp. 1-4, doi: 10.1109/ATEE.2013.6563419. keywords: {Heart rate variability;Electrocardiography;Atrial fibrillation;Cardiology;Databases;Computers;Measurement;atrial fibrillation prediction;surface ECG;HRV analysis;morphologic variability;decision rule},
        [3] Anwar, A., & Khammari, H. An Efficient Paroxysmal Atrial Fibrillation Prediction Method Using CWT and SVM,
        [4] Buś, S., Jędrzejewski, K., & Guzik, P. (2022). Statistical and Diagnostic Properties of pRRx Parameters in Atrial Fibrillation Detection. Journal of Clinical Medicine, 11(19), 5702. https://doi.org/10.3390/jcm11195702, 
        [5] Parsi, A., Glavin, M., Jones, E., & Byrne, D. (2021). Prediction of paroxysmal atrial fibrillation using new heart rate variability features. Computers in biology and medicine, 133, 104367. https://doi.org/10.1016/j.compbiomed.2021.104367
        [6] Gavidia, M., Zhu, H., Montanari, A. N., Fuentes, J., Cheng, C., Dubner, S., Chames, M., Maison-Blanche, P., Rahman, M. M., Sassi, R., Badilini, F., Jiang, Y., Zhang, S., Zhang, H. T., Du, H., Teng, B., Yuan, Y., Wan, G., Tang, Z., He, X., … Goncalves, J. (2024). Early warning of atrial fibrillation using deep learning. Patterns (New York, N.Y.), 5(6), 100970. https://doi.org/10.1016/j.patter.2024.100970
        [7] C. Maier, M. Bauch and H. Dickhaus, "Screening and prediction of paroxysmal atrial fibrillation by analysis of heart rate variability parameters," Computers in Cardiology 2001. Vol.28 (Cat. No.01CH37287), Rotterdam, Netherlands, 2001, pp. 129-132, doi: 10.1109/CIC.2001.977608. keywords: {Atrial fibrillation;Heart rate variability;Testing;Electrocardiography;Performance analysis;Sampling methods;Heart rate detection;Heart rate;Fluctuations;Polynomials},
        [8] Grégoire, J. M., Gilon, C., Marelli, F., Godart, P., Bersini, H., & Carlier, S. (2025). Autonomic Nervous System Activity before Atrial Fibrillation Onset as Assessed by Heart Rate Variability. Reviews in cardiovascular medicine, 26(1), 25364. https://doi.org/10.31083/RCM25364
        [9] Ebrahimzadeh, E., Kalantari, M., Joulani, M., Shahraki, R. S., Fayaz, F., & Ahmadi, F. (2018). Prediction of paroxysmal Atrial Fibrillation: A machine learning based approach using combined feature vector and mixture of expert classification on HRV signal. Computer methods and programs in biomedicine, 165, 53–67. https://doi.org/10.1016/j.cmpb.2018.07.014
        [10] Shin, D. G., Yoo, C. S., Yi, S. H., Bae, J. H., Kim, Y. J., Park, J. S., & Hong, G. R. (2006). Prediction of paroxysmal atrial fibrillation using nonlinear analysis of the R-R interval dynamics before the spontaneous onset of atrial fibrillation. Circulation journal : official journal of the Japanese Circulation Society, 70(1), 94–99. https://doi.org/10.1253/circj.70.94
        [11] Zhu, H., Jiang, N., Xia, S., & Tong, J. (2024). Atrial Fibrillation Prediction Based on Recurrence Plot and ResNet. Sensors, 24(15), 4978. https://doi.org/10.3390/s24154978

        """
        return {
            "HRV_MeanNN": "The mean of the RR intervals. [1-4]", 
            "HRV_SDNN": "The standard deviation of the RR intervals.[2]",
            "HRV_SDANN1, HRV_SDANN2, HRV_SDANN5[2] ": """The standard 
            deviation of average RR intervals extracted from n-minute 
            segments of time series data (1, 2 and 5 by default). 
            Note that these indices require a minimal duration of signal to be computed (3, 6 and 15 minutes respectively) 
            and will be silently skipped if the data provided is too short.""",
            "HRV_RMSSD": """The square root of the mean of the squared successive differences between  adjacent RR intervals.[1]""",
            "HRV_SDSD" : "The standard deviation of the successive differences between RR intervals.[7]",
            "HRV_CVNN" : "The standard deviation of the RR intervals (**SDNN**) divided by the mean of the RR intervals (**MeanNN**).[6]",
            "HRV_CVSD" : "The root mean square of successive differences (**RMSSD**) divided by the mean of the RR intervals (**MeanNN**).",
            "HRV_MedianNN": "The median of the RR intervals.",
            "HRV_MadNN": "The median absolute deviation of the RR intervals.",
            "HRV_MCVNN": "The median absolute deviation of the RR intervals (**MadNN**) divided by the median of the RR intervals (**MedianNN**).",
            "HRV_IQRNN": "The interquartile range (**IQR**) of the RR intervals.",
            "HRV_SDRMSSD": "SDNN / RMSSD, a time-domain equivalent for the low Frequency-to-High Frequency (LF/HF) Ratio (Sollers et al., 2007).",
            "HRV_Prc20NN": "The 20th percentile of the RR intervals ",
            "HRV_Prc80NN": "The 80th percentile of the RR intervals ",
            "HRV_pNN50": "The percentage of absolute differences in successive RR intervals greater than 50 ms [1]",
            "HRV_pNN20": "The percentage of absolute differences in successive RR intervals greater than 20 ms (Mietus et al., 2002).[1]",
            "HRV_MinNN": "The minimum of the RR intervals (Parent, 2019; Subramaniam, 2022).",
            "HRV_MaxNN": "The maximum of the RR intervals (Parent, 2019; Subramaniam, 2022).",
            "HRV_TINN": """A geometrical parameter of the HRV, or more specifically, the baseline width of
          the RR intervals distribution obtained by triangular interpolation, where the error of
          least squares determines the triangle. It is an approximation of the RR interval
          distribution.""",
            "HRV_HTI": """The HRV triangular index, measuring the total number of RR intervals divided by
          the height of the RR intervals histogram.""", 
            "NN50": """The number of pairs of successive R-R intervals that differ by more than 50 ms.[5]""",
            "pNN31": """The percentage of successive intervals differing by at least 31 ms. [4]""",
            "Mean_HR": "The mean heart rate in beats per minute (bpm).[1]",
            "SD_HR": "The standard deviation of the heart rate in bpm.[1]",
            "Min_HR": "The minimum heart rate in bpm.[1]",
            "Max_HR": "The maximum heart rate in bpm.[1]",
            "HRV_ULF": """The spectral power of ultra low frequencies (by default, .0 to .0033 Hz). Very long signals are required for this to index to be
                        extracted, otherwise, will return NaN.""",
            "HRV_VLF": """The spectral power of very low frequencies (by default, .0033 to .04 Hz). [8]""",
            "HRV_LF": """The spectral power of low frequencies (by default, .04 to .15 Hz).[9]""",
            "HRV_HF": """The spectral power of high frequencies (by default, .15 to .4 Hz).[5]""",
            "HRV_VHF": """The spectral power of very high frequencies (by default, .4 to .5 Hz).""",
            "HRV_TP": """The total spectral power.""",
            "HRV_LFHF": """The ratio obtained by dividing the low frequency power by the high frequency power.""",
            "HRV_LFn": """The normalized low frequency, obtained by dividing the low frequency power by the total power.""",
            "HRV_HFn": """The normalized high frequency, obtained by dividing the high frequency power by the total power.""",
            "HRV_LnHF": """The log transformed HF.""", 
            "HRV_SD1": (
                "Standard deviation perpendicular to the Poincaré plot line of identity. "
                "Index of short-term (beat-to-beat) RR variability. Equivalent to RMSSD. [1]"
            ),
            "HRV_SD2": (
                "Standard deviation along the Poincaré plot line of identity. "
                "Index of long-term HRV changes. [1]"
            ),
            "HRV_SD1SD2": "Ratio SD1 / SD2. Describes short-term to long-term variability balance. [5]",
            "HRV_S": (
                "Area of the Poincaré plot ellipse (π × SD1 × SD2). "
                "Proportional to SD1SD2; overall HRV measure. "
            ),
            "HRV_CSI": (
                "Cardiac Sympathetic Index (Toichi, 1997). Longitudinal variability (4×SD2) "
                "divided by transverse variability (4×SD1). Independent of vagal activity. "
            ),
            "HRV_CVI": (
                "Cardiac Vagal Index (Toichi, 1997). Logarithm of the product of "
                "longitudinal (4×SD2) and transverse (4×SD1) variability. "
                "Unaffected by sympathetic activity. "
            ),
            "HRV_CSI_Modified": (
                "Modified CSI (Jeppesen, 2014). Square of longitudinal variability "
                "divided by transverse variability. "
            ),

            # ── Heart Rate Asymmetry (Poincaré) ───────────────────────────────
            "HRV_GI": (
                "Guzik's Index. Distance of Poincaré points above the line of identity (LI) "
                "to LI, divided by distance of all off-LI points to LI. "
            ),
            "HRV_SI": (
                "Slope Index. Phase angle of points above LI divided by phase angle "
                "of all off-LI points. "
            ),
            "HRV_AI": (
                "Area Index. Cumulative sector area of points above LI divided by "
                "cumulative sector area of all off-LI points. "
            ),
            "HRV_PI": (
                "Porta's Index. Number of Poincaré points below LI divided by "
                "total off-LI points. "
            ),
            "HRV_SD1d": (
                "Short-term variance of contributions of decelerations "
                "(prolongations of RR intervals) to HRV.  "
            ),
            "HRV_SD1a": (
                "Short-term variance of contributions of accelerations "
                "(shortenings of RR intervals) to HRV.  "
            ),
            "HRV_C1d": "Contribution of heart rate decelerations to short-term HRV.  ",
            "HRV_C1a": "Contribution of heart rate accelerations to short-term HRV.  ",
            "HRV_SD2d": (
                "Long-term variance of contributions of decelerations "
                "(prolongations of RR intervals) to HRV. ) "
            ),
            "HRV_SD2a": (
                "Long-term variance of contributions of accelerations "
                "(shortenings of RR intervals) to HRV.  "
            ),
            "HRV_C2d": "Contribution of heart rate decelerations to long-term HRV.  ",
            "HRV_C2a": "Contribution of heart rate accelerations to long-term HRV.  ",
            "HRV_SDNNd": "Total variance of contributions of decelerations to HRV.  ",
            "HRV_SDNNa": "Total variance of contributions of accelerations to HRV.  ",
            "HRV_Cd": "Total contribution of heart rate decelerations to HRV.  ",
            "HRV_Ca": "Total contribution of heart rate accelerations to HRV.  ",

            # ── Heart Rate Fragmentation ───────────────────────────────────────
            "HRV_PIP": "Percentage of inflection points in the RR interval series.  ",
            "HRV_IALS": (
                "Inverse of the average length of acceleration/deceleration segments. "
                "Higher values indicate greater fragmentation.  "
            ),
            "HRV_PSS": "Percentage of short segments in the RR interval series.  ",
            "HRV_PAS": "Percentage of NN intervals in alternation segments.  ",

            # ── Detrended Fluctuation Analysis ────────────────────────────────
            "HRV_DFA_alpha1": (
                "Short-term scaling exponent from monofractal DFA (n ≈ 4–16 beats). "
                "α1 < 0.5: anti-correlated; α1 ≈ 1.0: 1/f noise; α1 > 1.5: non-stationary. [1]"
            ),
            "HRV_DFA_alpha2": (
                "Long-term scaling exponent from monofractal DFA (n ≈ 16–64 beats). "
                "Reflects long-range correlations. [8]"
            ),

            # ── Multifractal DFA — alpha1 scale ───────────────────────────────
            "HRV_MFDFA_alpha1_Width":      "Width of the multifractal spectrum at the alpha1 scale. Larger values indicate greater multifractality. ",
            "HRV_MFDFA_alpha1_Peak":       "Peak (most probable) singularity exponent of the multifractal spectrum at the alpha1 scale. ",
            "HRV_MFDFA_alpha1_Mean":       "Mean singularity exponent of the multifractal spectrum at the alpha1 scale.  ",
            "HRV_MFDFA_alpha1_Max":        "Maximum singularity exponent of the multifractal spectrum at the alpha1 scale.  ",
            "HRV_MFDFA_alpha1_Delta":      "Asymmetry index of the multifractal spectrum at the alpha1 scale (right minus left width).  ",
            "HRV_MFDFA_alpha1_Asymmetry":  "Normalised asymmetry of the multifractal spectrum at the alpha1 scale.  ",
            "HRV_MFDFA_alpha1_Fluctuation":"Fluctuation of the generalised Hurst exponent at the alpha1 scale.  ",
            "HRV_MFDFA_alpha1_Increment":  "Increment of the multifractal spectrum at the alpha1 scale.  ",

            # ── Multifractal DFA — alpha2 scale ───────────────────────────────
            "HRV_MFDFA_alpha2_Width":      "Width of the multifractal spectrum at the alpha2 scale.  ",
            "HRV_MFDFA_alpha2_Peak":       "Peak singularity exponent of the multifractal spectrum at the alpha2 scale.  ",
            "HRV_MFDFA_alpha2_Mean":       "Mean singularity exponent of the multifractal spectrum at the alpha2 scale.  ",
            "HRV_MFDFA_alpha2_Max":        "Maximum singularity exponent of the multifractal spectrum at the alpha2 scale.  ",
            "HRV_MFDFA_alpha2_Delta":      "Asymmetry index of the multifractal spectrum at the alpha2 scale.  ",
            "HRV_MFDFA_alpha2_Asymmetry":  "Normalised asymmetry of the multifractal spectrum at the alpha2 scale.  ",
            "HRV_MFDFA_alpha2_Fluctuation":"Fluctuation of the generalised Hurst exponent at the alpha2 scale.  ",
            "HRV_MFDFA_alpha2_Increment":  "Increment of the multifractal spectrum at the alpha2 scale.  ",

            # ── Entropy metrics ────────────────────────────────────────────────
            "HRV_ApEn": (
                "Approximate Entropy. Quantifies regularity/predictability of the RR series; "
                "lower values indicate more regular dynamics. [10]"
            ),
            "HRV_SampEn": (
                "Sample Entropy. Similar to ApEn but less biased; "
                "lower values indicate more regular dynamics. [10]"
            ),
            "HRV_ShanEn": (
                "Shannon Entropy of the RR interval distribution. "
                "Measures information content of the series. [5]"
            ),
            "HRV_FuzzyEn": (
                "Fuzzy Entropy. Uses fuzzy membership functions to assess regularity; "
                "more robust to noise than SampEn. "
            ),
            "HRV_MSEn": (
                "Multiscale Entropy. Assesses complexity across multiple temporal scales. "
                "Healthy dynamics tend to have higher MSEn across scales. "
            ),
            "HRV_CMSEn": "Composite Multiscale Entropy. Reduces variance of MSEn estimates. ",
            "HRV_RCMSEn": "Refined Composite Multiscale Entropy. Further improves CMSEn accuracy. ",

            # ── Fractal / complexity metrics ───────────────────────────────────
            "HRV_CD": (
                "Correlation Dimension. Estimates the fractal dimension of the RR attractor; "
                "lower values indicate less complex dynamics. "
            ),
            "HRV_HFD": (
                "Higuchi Fractal Dimension. Measures signal complexity directly from the time series; "
                "values range from 1 (simple) to 2 (highly complex). "
            ),
            "HRV_KFD": (
                "Katz Fractal Dimension. Estimates fractal dimension from waveform diameter "
                "and total length. [8]"
            ),
            "HRV_LZC": (
                "Lempel-Ziv Complexity. Measures algorithmic complexity (compressibility) "
                "of the binarised RR series. [8]"
            ),

            # ══════════════════════════════════════════════════════════════════
            # CUSTOM RQA DOMAIN
            # ══════════════════════════════════════════════════════════════════

            # ── Eigenvalue-based (phase-space covariance) ──────────────────────
            "NME": (
                "Normalised Maximum Eigenvalue. Ratio of the largest eigenvalue of the "
                "phase-space covariance matrix to the total variance. High NME indicates "
                "that most variance lies along a single direction (low complexity). [7]"
            ),
            "EEV": (
                "Eigenvalue Entropy. Shannon entropy of the normalised eigenvalue spectrum. "
                "Low values indicate anisotropic (low-dimensional) phase-space structure. [7]"
            ),

            # ── Diagonal line features ────────────────────────────────────────
            "RecurrenceRate_RRFixed": (
                "Recurrence Rate (custom, fixed-RR threshold). Fraction of recurrent points "
                "in the recurrence matrix (excluding the main diagonal). [11] "
            ),
            "Determinism_RRFixed": (
                "Determinism (custom). Fraction of recurrent points forming diagonal lines "
                "of length ≥ l_min. Higher values indicate more predictable dynamics. [11]"
            ),
            "L_RRFixed": (
                "Mean diagonal line length (custom). Average length of diagonal structures "
                "≥ l_min; related to the predictability time of the system. [11]"
            ),
            "LMax_RRFixed": "Maximum diagonal line length (custom). [11]",
            "LMedian_RRFixed": "Median diagonal line length (custom). ",
            "LEn_RRFixed": (
                "Diagonal line length entropy (custom). Shannon entropy of the diagonal "
                "line length distribution; higher values indicate more complex line structures."
            ),
            "Divergence_RRFixed": (
                "Divergence (custom). Reciprocal of LMax (1/LMax). Proxy for the largest "
                "positive Lyapunov exponent; higher values indicate faster divergence. "
            ),
            "Ratio_RRFixed": "DET / RR ratio (custom). ",
            "LMinCount_RRFixed": "Total count of diagonal lines with length ≥ l_min (custom). ",

            # ── Vertical line features ────────────────────────────────────────
            "Laminarity_RRFixed": (
                "Laminarity (custom). Fraction of recurrent points forming vertical lines "
                "of length ≥ v_min. Reflects intermittent laminar states. [11]"
            ),
            "TrappingTime_RRFixed": (
                "Trapping Time (custom). Mean vertical line length ≥ v_min. "
                "Estimates the average time the system remains in a laminar state. [11]"
            ),
            "VMaX_RRFixed": "Maximum vertical line length (custom). ",
            "VEn_RRFixed": (
                "Vertical line length entropy (custom). Shannon entropy of the vertical "
                "line length distribution. "
            ),

            # ── White vertical line features ──────────────────────────────────
            "W_RRFixed": (
                "Mean white vertical line length (custom). Average time between recurrences "
                "(recurrence time); related to the inverse of recurrence rate. "
            ),
            "WMax_RRFixed": "Maximum white vertical line length (custom). ",

            # ── Advanced ──────────────────────────────────────────────────────
            "K2": (
                "Rényi entropy of second order (K2) estimated from the slope of log Pc(l) "
                "vs diagonal line length l. Provides a lower bound for the Kolmogorov-Sinai "
                "entropy; estimated here by linear regression over the scaling region. [14]"
            ),

            # ══════════════════════════════════════════════════════════════════
            # NEUROKIT2 RQA DOMAIN  (hrv_rqa)
            # ══════════════════════════════════════════════════════════════════

            "RecurrenceRate": (
                "Recurrence Rate (NeuroKit2). Percentage of recurrent points in the "
                "recurrence plot (excluding the main diagonal). [11]"
            ),
            "DiagRec": (
                "Diagonal Recurrence (NeuroKit2). Recurrence rate computed from "
                "diagonal line structures only. "
            ),
            "Determinism": (
                "Determinism (NeuroKit2). Fraction of recurrent points belonging to "
                "diagonal lines of length ≥ l_min. [11]"
            ),
            "DeteRec": (
                "DET / RR ratio (NeuroKit2). Ratio of Determinism to Recurrence Rate; "
                "higher values indicate more structured recurrence."
            ),
            "L": "Mean diagonal line length (NeuroKit2). [11]",
            "Divergence": (
                "Divergence (NeuroKit2). Reciprocal of the maximum diagonal line length; "
                "proxy for the largest Lyapunov exponent. "
            ),
            "LEn": (
                "Diagonal line length entropy (NeuroKit2). Shannon entropy of the "
                "diagonal line length distribution. "
            ),
            "Laminarity": (
                "Laminarity (NeuroKit2). Fraction of recurrent points forming vertical "
                "lines of length ≥ v_min. [11]"
            ),
            "TrappingTime": (
                "Trapping Time (NeuroKit2). Mean vertical line length; average time "
                "the system spends in a laminar state. [111]"
            ),
            "VMax": "Maximum vertical line length (NeuroKit2). [11]",
            "VEn": (
                "Vertical line length entropy (NeuroKit2). Shannon entropy of the "
                "vertical line length distribution. "
            ),
            "W": (
                "Mean white vertical line length (NeuroKit2). Average recurrence time "
                "(time between consecutive recurrences in the same column). "
            ),
            "WMax": "Maximum white vertical line length (NeuroKit2).",
            "WEn": (
                "White vertical line length entropy (NeuroKit2). Shannon entropy of the "
                "distribution of white vertical line lengths (recurrence time distribution). "
            )
        }
    
    
    
