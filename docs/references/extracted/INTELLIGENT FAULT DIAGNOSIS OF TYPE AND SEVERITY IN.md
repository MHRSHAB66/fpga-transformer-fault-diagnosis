# INTELLIGENT FAULT DIAGNOSIS OF TYPE AND SEVERITY IN LOW-FREQUENCY, LOW BIT-DEPTH SIGNALS 

A PREPRINT 

**Tito Spadini** 

**Tito Spadini Kenji Nose-Filho** Laboratório de Sinais e Sistemas (LSS) Laboratório de Sinais e Sistemas (LSS) Universidade Federal do ABC (UFABC) Universidade Federal do ABC (UFABC) Santo André, SP, Brazil Santo André, SP, Brazil `tito.caco@ufabc.edu.br kenji.nose@ufabc.edu.br` 

## **Ricardo Suyama** 

Laboratório de Sinais e Sistemas (LSS) Universidade Federal do ABC (UFABC) Santo André, SP, Brazil `ricardo.suyama@ufabc.edu.br` 

## **ABSTRACT** 

This study focuses on Intelligent Fault Diagnosis (IFD) in rotating machinery utilizing a single microphone and a data-driven methodology, effectively diagnosing 42 classes of fault types and severities. The research leverages sound data from the imbalanced MaFaulDa dataset, aiming to strike a balance between high performance and low resource consumption. The testing phase encompassed a variety of configurations, including sampling, quantization, signal normalization, silence removal, Wiener filtering, data scaling, windowing, augmentation, and classifier tuning using XGBoost. Through the analysis of time, frequency, mel-frequency, and statistical features, we achieved an impressive accuracy of 99.54% and an F-Beta score of 99.52% with just 6 boosting trees at an 8 kHz, 8-bit configuration. Moreover, when utilizing only MFCCs along with their first- and second-order deltas, we recorded an accuracy of 97.83% and an F-Beta score of 97.67%. Lastly, by implementing a greedy wrapper approach, we obtained a remarkable accuracy of 96.82% and an F-Beta score of 98.86% using 50 selected features, nearly all of which were first- and second-order deltas of the MFCCs. 

_**Keywords**_ Machine Fault Diagnosis _·_ Intelligent Fault Diagnosis _·_ Acoustic Fault Classification _·_ Sound Event Recognition 

## **List of Acronyms and Abbreviations** 

ADASYN Adaptive Synthetic ANN Artificial Neural Network APF-KNN Asymmetric Proximity Function K-Nearest Neighbor ASFE Adaptive Selection Feature Engineering BPN Back-Propagation Network 

This work was financed in part by the Coordenação de Aperfeiçoamento de Pessoal de Nível Superior - Brasil (CAPES) - Finance Code 001, the National Council for Scientific and Technological Development (CNPq) grant #311380/2021-2, and the São Paulo Research Foundation (FAPESP) grant #2020/09838-0 (BI0S - Brazilian Institute of Data Science). 

_IFD of Type and Severity in Low-Frequency, Low Bit-depth Signals_ 

A PREPRINT 

|CBR|Case-Based Reasoning|
|---|---|
|CSV|Comma Separated Values|
|CWRU|Case Western Reserve University|
|CWT|Continuous Wavelet Transform|
|dBFS|decibel Full-Scale|
|DSP|Digital Signal Processing|
|DT|Decision Tree|
|DWT|Discrete Wavelet Transform|
|EMD|Empirical Mode Decomposition|
|ENN|Extension Neural Network|
|FCF|Fault Characteristic Frequency|
|FFT|Fast Fourier Transform|
|FSL|Few-Shot Learning|
|GRNN|Generalized Regression Neural Network|
|HATS|Head and Torso Simulator|
|HHT|Hilbert-Huang Transform|
|ICE|Internal Combustion Engines|
|IFD|Intelligent Fault Diagnosis|
|IMF|Intrinsic Mode Function|
|KNN|K-Nearest Neighbor|
|LDA|Linear Discriminant Analysis|
|LSSVM|Least Square Support Vector Machine|
|MaFaulDa|Machinery Fault Database|
|MFCC|Mel-Frequency Cepstral Coeffcients|
|MFE-SVM|Multidomain Feature Extraction with Support Vector Machine|
|MFPE|Multiscale Fraction Permutation Entropy|
|NB|Naive Bayes|
|NN|Nearest Neighbor|
|NOSW|Non-Overlapping Sliding Window|
|PCA|Principal Component Analysis|
|PSD|Power Spectrum Density|
|PSO|Particle Swarm Optimization|
|RBFNN|Radial Basis Function Neural Network|
|RF|Random Forest|
|RMS|Root Mean Square|
|RWE|Relative Wavelet Energy|
|SBM|Similarity-Based Modeling|
|SFSFT|Stepping Frane Synchrosqueezed Fourier Transform|
|SMOTE|Synthetic Minority Over-sampling Technique|
|SOM|Self-Organizing Map|
|STFT|Short-Time Fourier Transform|
|SVM|Support Vector Machine|
|TBM|Transferable Belief Model|
|TSAAR|Transient-based Angular Resampling|



2 

_IFD of Type and Severity in Low-Frequency, Low Bit-depth Signals_ 

A PREPRINT 

TSFDR-LDA Time-Spectral Feature Extraction and Reduction with Linear Discriminant Analysis VGG Visual Geometry Group WG Worm Gearbox WPA Wavelet Packet Analysis WPD Wavelet Packet Decomposition WPDE Wave Packet Decomposition Energy WPT Wavelet Packet Transform XGBoost eXtreme Gradient Boosting 

## **1 Introduction** 

In various industrial sectors, mechanical and electromechanical machines are composed of multiple components, each susceptible to failure. Failures can affect the entire system, individual parts, or specific components. Some machines operate intermittently, while others require continuous operation, often in harsh environments that are detrimental to both human operators and the machines themselves. Under these conditions, failure is not a matter of “if” but “when”; given sufficient time, it becomes nearly inevitable. 

**Figure 1:** Relationship between availability and manufacturer’s cost. Based on [1]. 

**==> picture [250 x 205] intentionally omitted <==**

**----- Start of picture text -----**<br>
Costs Costs<br>after before<br>delivery Total delivery<br>SZ<br>Availability<br>Manufacturer's Cost<br>**----- End of picture text -----**<br>


The inevitability of machine failures raises significant safety and economic concerns. Failures can lead to severe accidents or fatalities, while economically, they result in financial losses and disrupt socio-economic structures dependent on machine operation. 

Figure 1, adapted from [1] (p. 32), illustrates the relationship between machine availability and maintenance costs, highlighting implications for manufacturers and stakeholders. From the manufacturer’s perspective, evaluating predelivery and post-delivery periods by operational availability shows that higher availability incurs increased pre-delivery costs but reduces post-delivery costs. 

Achieving this balance requires stringent design standards, high-quality materials, skilled labor, and rigorous testing, leading to fewer failures and safeguarding the company’s reputation. However, striving for high availability can escalate costs, necessitating optimized quality within practical limits. This work emphasizes health and well-being, asserting that economic factors, while significant, cannot overshadow the priority of human lives. 

Maintaining machine health is crucial, involving preventive and corrective maintenance strategies. Preventive maintenance seeks to avert failures, while corrective maintenance addresses repairs post-failure. A diagnostic stage is critical for fault classification and severity assessment as noted in [2] (p. 4). 

3 

_IFD of Type and Severity in Low-Frequency, Low Bit-depth Signals_ 

A PREPRINT 

A key component of Industry 4.0 is implementing data-driven solutions like the Internet of Things (IoT), Cloud Computing, Digital Signal Processing (DSP), Big Data, and machine learning. These technologies require substantial data; while Few-Shot Learning (FSL) targets effectiveness with minimal data, most algorithms depend on extensive datasets. This demand for data has led to a focus on populating large databases, which often lack refined information, complicating practical applications. 

Sensors collect data on machines and environments, enabling diagnostic methods like Intelligent Fault Diagnosis (IFD) [3]. Sound alone can provide significant latent information for accurate classification through signal processing techniques. The effectiveness of sound characteristics depends on machine architecture, operating conditions, and environmental factors, yet many low-cost microphones meet monitoring needs adequately. 

The audio signal processing field is robust, equipped with tools for visualizing sound through waveforms and spectrograms. Human hearing also serves as an auditing mechanism. Established methods exist for effective noise treatment and advanced source separation in noisy environments. 

Microphones are primary devices for sound data collection, available in various designs and sizes. Typically low-cost, they capture significant data quickly and can be positioned far from the sound source. Depending on application, microphones can be directional, like Shotgun types, or omnidirectional, which capture sound uniformly. Their design flexibility allows for tailored configurations, while their minimally invasive nature enables simple, cost-effective installations with minimal modifications. 

Using sound to diagnose machine failures enhances diagnostic capabilities, ensuring reliable monitoring even when traditional sensors are impractical or compromised. 

Therefore, sound signal information can be effectively used to diagnose machinery failures. As noted by Lei et al. [3] in their discussion about future challenges and a roadmap in IFD, combining resampling with ensemble methods like XGBoost can effectively address common issues in this field, such as data imbalance. 

This work is organized as follows: Section 2 reviews related studies, summarizing key information in a table. Section 3 describes the dataset’s characteristics. Section 4 details data preparation steps, including signal conversion, normalization, silence removal, denoising, feature extraction, and scaling. Section 5 explains the data division for training and testing. Section 6 discusses data augmentation to address class imbalance. Section 7 covers the classification process, cross-validation, and performance metrics. Section 8 outlines the workflow with step-by-step illustrations and simulation environment details. Section 9 presents results, analyzing parameter impacts and summarizing key configurations and features. Section 10 presents our conclusions and final considerations regarding this work. 

## **2 Related works** 

There are many different scenarios for working with machine fault diagnosis, but this study focuses solely on scenarios where the only source of information is sound. However, it is not just any scenario involving sound; it is one in which a single microphone captures all the ambient noise, forming a mixture that results in a one-dimensional signal. Nevertheless, there are studies that, in addition to using sound, also incorporate various other sensors besides the microphone to gather additional sources of information. In this case, for comparative purposes regarding what can be achieved using sound alone, it might be interesting to consider them here. 

The use of sound to diagnose machinery faults is not a new concept. Lee and White [4] applied a two-stage adaptive filtering process to enhance impulsive noise, then analyzed the result in the time-frequency domain for fault detection in rotating machinery. Shibata et al. [5] proposed a method for visualizing machinery sound and used time-frequency analysis and wavelet analysis to identify fault-related components. Lin [6] emphasized sound signal denoising through wavelets for improved classification. While these works do not focus on machine learning-based classification, they contribute to diagnostic tasks through classical approaches. 

Benko et al. [7] designed an environment to record motor sounds in a small anechoic chamber, focusing on sound signal processing and noise treatment. They used Root Mean Square (RMS) of the sound signal, Power Spectrum Density (PSD), Short-Time Fourier Transform (STFT), and Hilbert Transform for analysis. No classification was performed; instead, diagnosis was made using an incidence matrix and the Transferable Belief Model (TBM). The data was not made available, preventing further analysis or comparison with other approaches. 

Benko et al. [8] presents a fault diagnosis system for vacuum cleaner motors using sound analysis with techniques such as Hilbert Transform, amplitude demodulation, Butterworth filters, PSD analysis, and weighted histograms. These techniques identify faults such as bearing wear, intermittent rubbing, and issues with brush-commutator contact. The system efficiently detects unusual sounds generated by fan impeller deformations and outperforms methods like current and vibration analysis, which proved less effective for these motors. 

4 

_IFD of Type and Severity in Low-Frequency, Low Bit-depth Signals_ 

A PREPRINT 

Olsson et al. [9] conducted experiments for fault diagnosis in industrial robot assembly, recording sound for various cases. Despite some details provided, such as the use of 10 robots with and without payloads and two fault types, critical information about the experimental setup, sound sources, and microphones is missing. The data was not made publicly available, preventing verification and analysis. Noise filtering was applied using Discrete Wavelet Transform (DWT), resulting in 64 features, and classification was done with a Case-Based Reasoning (CBR) approach and Nearest Neighbor (NN) classifier, achieving 99.1% similarity, but the lack of data access hinders result verification. 

Wu and Liu [10] proposed a fault diagnosis method for combustion engines using the entropy of sixteen Wavelet Packet Transform (WPT) coefficients from sound emissions as features. They considered six operating conditions. For each condition, they used 30 samples for training and 120 samples for testing. Their approach, leveraging a Generalized Regression Neural Network (GRNN) and a Back-Propagation Network (BPN), achieved an average accuracy of over 95%. 

Amarnath et al. [11] designed a custom sound recording environment with a microphone through various specific tests, but the data were not made publicly available, which prevents new classification proposals from being compared against their work. Nevertheless, it’s worth mentioning that their approach extracted statistical features from one-dimensional sound signals, and the classification was performed using a Decision Tree (DT) with 10-fold cross-validation. Their accuracy reached 95.5%, but it is impossible to assess how challenging it might have been to classify such data since they are unavailable. 

Pandya et al. [12] used acoustic signals for diagnosing bearing faults across five operating scenarios. They applied the Hilbert-Huang Transform (HHT) via Empirical Mode Decomposition (EMD) to extract Intrinsic Mode Functions (IMFs), from which nine statistical features were derived. Using an Asymmetric Proximity Function K-Nearest Neighbor (APFKNN) and 5-fold cross-validation on 180 samples (scaled from 0.1 to 0.9), they achieved approximately 96.67% accuracy. 

Germen et al. [13] created a sound recording rig using five cardioid microphones to record the motor sound, converting one-dimensional audio signals into 2D images through a [0, 255] normalization and Wavelet decomposition. They achieved nearly 100% classification accuracy using a Kohonen Self-Organizing Map (SOM) with the LVQ3 algorithm. However, the study lacks publicly available data for reproducibility, does not mention cross-validation, and has a relatively small sample size, raising concerns about potential bias and leaving the impressive results unverified. 

Shatnawi and Al-khassaweneh. [14] focuses on classifying seven fault classes in Internal Combustion Engines (ICE). Features were extracted using Wavelet Packet Decomposition (WPD), resulting in a 16-dimensional feature vector composed of energies and Shannon entropies of wavelet coefficients from various frequency bands. Two classifiers were employed: the Extension Neural Network (ENN), achieving up to 98.8% recognition rate, and the BPN, 91.5%. Dataset limitations include lack of public availability, limited operating conditions and fault types, simplicity of the addressed faults, and data collected from a single engine model (Lada 2107), potentially restricting generalization to other ICE models. 

Moosavian et al. [15] leveraged fused acoustic and vibration signals, denoised via wavelets, across three working conditions. They extracted 17 statistical features, applied 10-fold cross-validation, and used a 70/30 train-test split on 140 samples per signal type per condition. Combining a Least Square Support Vector Machine (LSSVM) and an Artificial Neural Network (ANN) achieved over 98.5% accuracy, with acoustic signals outperforming vibration when evaluated separately. 

Lu et al. [16] investigate fault identification in motor bearings using the Transient-based Angular Resampling (TSAAR) technique with data from experiments on bearings with artificial faults. Acoustic signals were recorded during speed fluctuations, encompassing three classes: outer raceway faults, inner raceway faults, and normal state. Preprocessing included signal segmentation and noise filtering, with features such as Fault Characteristic Frequency (FCF) and phase information. TSAAR achieved a Success Rate of 100% under low noise, decreasing to 15% in high noise; SNR ranged from 5.14 dB to -9.86 dB. Limitations include restricted public data access and lack of conventional classifiers for comparison. 

Madhusudana et al. [17] recorded ambient sound during face milling to classify a 4-class problem using 200 samples. Features were derived from DWT, Empirical Mode Decomposition (EMD), and statistical measures, with feature selection via DT (J48) and classification through ANN, Naive Bayes, K-Star, and SVM using 10-fold cross-validation. DWT features yielded the best results, with SVM achieving 83% accuracy. Subsequently, Madhusudana et al. [18], using the same dataset, achieved 81% accuracy by selecting only 8 DWT-based features and employing a Decision Tree classifier (J48) for classification. Despite detailing the sound acquisition setup, the data were not publicly shared, limiting verifiability and external comparison. 

5 

_IFD of Type and Severity in Low-Frequency, Low Bit-depth Signals_ 

A PREPRINT 

Therefore, although there are some important classic datasets in the field of machine fault diagnosis, such as the Case Western Reserve University (CWRU) Bearing Dataset [19], which is arguably the most widely known among them, few of these datasets include the type of data we seek. Given that this study relies solely on a one-dimensional sound source as its information input, including studies that use only other sources of information for machine fault diagnosis would not provide any meaningful contribution. 

Among the datasets found and evaluated, the only one that adequately met our needs is the Machinery Fault Database (MaFaulDa) [20], as, among the various sensors used in that study to collect machinery data during the trials, a microphone was employed to record the sound. More detailed information about the dataset will be provided in Section 3. Let us now review some of the studies that have used the MaFaulDa dataset and discuss their contributions. 

The study conducted by Ribeiro et al. [21] utilized the MaFaulDa and CWRU datasets, incorporating time-based and spectrum-based features. Additionally, similarity-based modeling (SBM) was employed to generate new feature sets, and classification was performed using a Random Forest (RF) classifier. However, this study utilized all the available sensors, which extends far beyond just sound, because, in the case of MaFaulDa, there are 8 different data sources, and only one of them is the sound. After training the model using a 10-fold cross-validation, the test accuracy reached 96.43% with the MaFaulDa dataset. 

Subsequently, the work conducted by Marins et al. [22], which involved the same team as the study [21], made further progress extending SBM to a multiclass model, using SBM as a standalone classifier, reducing the SBM computational complexity, adding new similarity metrics, and achieving an accuracy of 98.5%. 

Then, Rocha [23] employed different approaches regarding both features and classifiers. For feature extraction, they used RMS, Haar wavelet transform, fractal dimension, and Fourier transform with statistical data. As for classifiers, they explored K-Nearest Neighbors (KNN), Support Vector Machine (SVM), and eXtreme Gradient Boosting (XGBoost). After conducting the experiments, the best result was obtained using the Haar wavelet transform and the XGBoost classifier, achieving 98.7% accuracy after a 10-fold cross-validation. 

Wang et al. [24] used sound signals for fault diagnosis across nine operational categories. After preprocessing with Wavelet Packet Analysis (WPA), they applied an ANN for classification, achieving 100% accuracy at a specific evaluation point, demonstrating the effectiveness of their method. 

Sun et al. [25] presents a sound-based fault diagnosis method for railway point machines, specifically the ZDJ9 model, with data collected on-site in China. Preprocessing includes EMD for noise reduction, selecting components by kurtosis and energy. Features were extracted using Multiscale Fractional Permutation Entropy (MFPE) to capture signal complexity. Using a proprietary dataset and, complementarily, the CWRU vibration-based dataset, the evaluation split data 5:3 for training/testing holdout. The main classifier, a Particle Swarm Optimization (PSO)-optimized SVM, achieved 99% accuracy, outperforming 1-NN, RF, and Naive Bayes (NB). However, the proprietary nature of the sound data limits replicability and broader applicability. 

Altinors et al. [26] conducted sound-based fault diagnosis on a UAV across four operating conditions using three motors (1400 kV, 2200 kV, 2700 kV). Sound was recorded at 44.1 kHz, scaled to 0-1, and six statistical features were extracted. Using Decision Tree (DT), SVM with a quadratic kernel, and KNN ( _K_ = 10) with 10-fold cross-validation, they achieved notable accuracy on the 2200 kV motor: 99.16% with DT and 99.75% with both SVM and KNN. 

Połok and Bilski [27] performed fault diagnosis on a BMX bike ratchet system using acoustic signals (44.1 kHz, 16-bit). They extracted 32 temporal, frequency, and time-frequency features to classify five scenarios, applying Repeated Random Sub-Sampling with a 1:4 split for cross-validation. Using DT, KNN, and SVM classifiers, they achieved accuracies of 95%, 98.6%, and 98.4%, respectively. 

Cao et al. [28] recorded audio from a ZDJ9 Railway Point Machine under 10 conditions, obtaining 800 samples. Such data is proprietary and therefore cannot be accessed and used. After EMD preprocessing and extracting features from 15 IMFs, for each of the 15 IMFs, 10 time-based features, 4 frequency-based features and 8 Wave Packet Decomposition Energy Entropy (WPDE) features are extracted, that is, 22 features for each IMF, totaling 330 features, but dimensionality reduction narrowed the features to 18. Using various classifiers and ensembles, with a 5:3 training/test holdout, the best model achieved an average accuracy of 99–99.93% across trials, showing high diagnostic effectiveness. 

Suman et al. [29] used both sound and vibration signals for vehicle fault diagnosis. The acoustic signals were enhanced, filtered, and windowed, with 30 Mel-Frequency Cepstral Coefficients (MFCCs) extracted as features. However, no typical classification approach or detailed information on the data, method, or validation was provided. 

Zhu et al. [30] used acoustic signals to detect faults in a hydraulic piston pump. The signals were processed with Continuous Wavelet Transform (CWT), with five operational states and 1200 samples per state. The data were split 

6 

_IFD of Type and Severity in Low-Frequency, Low Bit-depth Signals_ 

A PREPRINT 

70% for training and 30% for testing. Classification was performed using a LeNet-type convolutional neural network optimized with PSO, achieving an average accuracy of 99.76%. 

Pacheco-Chérrez et al. [31] used acoustic signals to diagnose a system under four operating scenarios, employing three strategies: 1) Multidomain Feature Extraction with SVM (MFE-SVM) using Principal Component Analysis (PCA); 2) Relative Wavelet Energy (RWE) with RF; and 3) Time-Spectral Feature Extraction and Reduction with Linear Discriminant Analysis (TSFDR-LDA). Using 5-fold cross-validation, MFE-SVM achieved 96.78% accuracy, RWE 96.79%, and TSFDR-LDA 95.60%, with acoustic signals consistently outperforming vibration signals alone. 

Sha et al. [32] developed a framework using acoustic signals to detect cavitation across four flow levels. Data augmentation was applied with a Non-Overlapping Sliding Window (NOSW), followed by Fast Fourier Transform (FFT)based feature extraction, statistical feature extraction, and Adaptive Selection Feature Engineering (ASFE) for feature selection. Using an 80/20 train-test split, XGBoost achieved 91.67% accuracy. 

Karabacak et al. [33] tested Worm Gearboxes (WGs) using sensors to capture vibration, sound, and thermal images, extracting time and frequency-based features along with thermal images. They used a cubic kernel SVM and an ANN with 10 neurons in one hidden layer and 4 outputs, achieving 98.6% and 99.2% accuracy, respectively. For the ANN, cross-validation was done with 70% for training, 15% for validation, and 15% for testing, while for the SVM, it was 75% for training and 25% for validation. However, the data was not made available, and the differing validation methods raise concerns about robustness. 

Mian et al. [34] used a Head and Torso Simulator (HATS) with two microphones to capture bearing sounds under five fault types and three rotation speeds in a semi-anechoic chamber. Six sound quality features were extracted, and a quadratic kernel SVM classifier with 5-fold cross-validation achieved accuracies of 99.7% in low noise, 97% in moderate noise, and 90% in high noise conditions. The dataset is not publicly accessible. 

Das and Das [35] used the open MaFaulDa dataset, exploring the impact of nine different Wavelet transforms on classifier performance. The classifiers tested included RF, AdaBoost (C4.5), MLP with ReLU activation, and MLP with Tanh activation. A variety of time-based statistical features were extracted. The study employed 10-fold cross-validation, and using only the microphone as the data source, achieved an accuracy of 99.84% with the Haar Wavelet and RF classifier. 

Yue et al. [36] used vibration signals with Mel-scale spectrograms for bearing fault diagnosis, applying an SVM with a hybrid Gaussian-polynomial kernel, an Multilayer Perceptron (MLP), and comparing with Visual Geometry Group (VGG) and ResNet models. Across 16 tasks, with four classes and varying Mel coefficients (400, 800, 1200), the SVM and MLP models often achieved the best results. 

Mishra et al. [37] performed fault diagnosis on rotating machinery across different operating conditions, focusing here on acoustic signals. They analyzed eight setups involving a motor and bearings, testing various data imbalance scenarios. After preprocessing with Stepping Frame Synchrosqueezed Fourier Transform (SFSFT) and extracting 13 scaled statistical features, classification was done using a Radial Basis Function Neural Network (RBFNN) with 5-fold cross-validation, achieving accuracies from 99.45% to 100%. 

**Table 1:** Summary of Literature Review 

|**Ref.**|**Dataset**a|**Preprocessing **|**Features**|**Validation**|**Classes**|**Classifers**b|**Results**c|
|---|---|---|---|---|---|---|---|
|[9]|own|wavelet|wavelet|-|3|CBR + NN|99.1%|
|||denoising|coeffcients|||||
||||(total = 64)|||||
|[10]|own|WPT|wavelet|1:4|6|GRNN|95%|
||||coeffcients|training/test||||
||||(total = 16)|||||
|[11]|own|feature|mean and|10-fold|3|DT (C4.5)|95.5%|
|||selection|skewness|||||
||||(total = 2)|||||
|[12]|own|HHT, EMD,|statistical|5-fold|5|APF-KNN|96.67%|
|||IMF|(total = 9)|||||



7 

_IFD of Type and Severity in Low-Frequency, Low Bit-depth Signals_ 

A PREPRINT 

## (Continuation of Table 1) 

|**Ref.**|**Dataset**|**Preprocessing **|**Features**|**Validation**|**Classes**|**Classifers**|**Results**|
|---|---|---|---|---|---|---|---|
|[13]|own|wavelet|wavelet|-|5|SOM +|100%|
|||packet|coeffcients,|||LVQ3||
|||decomposition|cross-|||||
||||correlation|||||
||||coeffcients|||||
||||(total = 40)|||||
|[14]|own|wavelet|energy, entropy|-|5 and 7d|ENN|98.8%|
|||packet|of|||||
|||wavelet|(total = 32)|||BPN|91.5%|
|||coeffcients||||||
|||decomposition||||||
|[15]|own|wavelet|statistical|10-fold|3|SVM,|98.5%|
|||denoising|(total = 17)|||ANN||
|[16]|own|impulse|FCF, phase|-|3|TSAAR|15–100%|
|||windowing,|information,|||||
|||noise fltering|signal envelope|||||
||||(total = ?)|||||
|[21]|MaFaulDa|-|time-based,|10-fold|6|SBM + RF|96.4%|
||||frequency-|||||
||||based,|||||
||||rotation|||||
||||frequency|||||
||||(total = 16)|||||
|[22]|MaFaulDa|-|time-based,|10-fold|10e|SBM|98.5%|
||||frequency-|||||
||||based,|||||
||||rotation|||||
||||frequency|||||
||||(total = 46)|||||
|[23]|MaFaulDa|dimensionality|RMS-based,|10-fold|6|KNN|91.8%|
|||reduction|Haar wavelet,|||SVM|96.9%|
||||fractal|||XGBoost|98.7%|
||||dimension,|||||
||||Fourier|||||
||||transform-based|||||
||||with statistics|||||
||||(total = 46)|||||
|[25]|own|EMD|MFPE|5:3|10|PSO + SVM|99.0%|
|||denoising,|(total = ?)|training/test||1-NN|98.6%|
|||IMF||||RF|95.3%|
|||||||NB|94.67%|
|[28]|own|EMD,|time-based,|5:3|10|ensembleg|99–99.93%|
|||IMF, dimen-|frequency-|training/test||||
|||sionality|based,|||||
|||reduction||||||



8 

_IFD of Type and Severity in Low-Frequency, Low Bit-depth Signals_ 

A PREPRINT 

## (Continuation of Table 1) 

|**Ref.**|**Dataset**|**Preprocessing **|**Features**|**Validation**|**Classes**|**Classifers**|**Results**|
|---|---|---|---|---|---|---|---|
||||WPDE|||||
||||(total = 18)|||||
|[24]|own|WPA|wavelet|1:1:1 train,|9|ANN|100%|
||||coeffcients|validation,||||
||||(total =21_×_9)|test||||
|[33]|own|-|time-based,|10-fold|4|SVM|98.6%|
||||frequency-|||ANN|99.2%|
||||based,|||||
||||thermal images|||||
||||(total = 19)|||||
|[26]|own|scaling|statistical|10-fold|4|DT|99.16%|
||||(total = 6)|||SVM|99.75%|
|||||||KNN|99.75%|
|[27]|own|normalization|time,|1:4|5|DT|95%|
||||frequency,|training/test||KNN|98.6%|
||||time-frequency|||SVM|98.4%|
||||(total = 32)|||||
|[32]|own|NOSW,|statistical|80%/20%|4|XGBoost|91.67%|
|||FFT|(total = ?)|training/test||||
|[30]|own|CWT|wavelet|70%/30%|5|PSO-LeNet|99.76%|
||||coeffcients|training/test||||
||||(total = ?)|||||
|[31]|own|RWE|wavelet|5-fold|4|RF|96.78%|
||||coeffcients|||||
||||(total = 9)|||||
|[34]|own|-|sound quality|5-fold|5|SVM|90–99.7%|
||||metrics|||(quadratic)||
||||(total = 6)|||||
|[35]|MaFaulDa|wavelet|time-based|10-fold|10|RF|99.84%f|
|||fltering|statistics|||||
||||(total = 12)|||||
|[37]|own|SFSFT|statistical|5-fold|8|RBFNN|99.45–|
||||||||100%|
||||(total = 13)|||||



> a Some of the studies used more than one dataset, but this table only considers datasets that have utilized sound. 

> b Some studies do not use an approach based on classical classification, but rather a more direct diagnosis based on some criteria. 

> c Some works have large amounts of results; here, we focus on bringing only the main results that have used sound as a source of information. 

> d This work used two cars. For one of them, they used 5 classes; for the other one, 7 classes. 

> e Although the same dataset was used, the work began to differentiate the types of overhang and underhang for each of the parts: ball, cage and outer race. 

> f Best result obtained using sound. 

> g Ensemble composed of SVM + NB + 1NN + 3NN + 5NN + LDA + DT. 

9 

_IFD of Type and Severity in Low-Frequency, Low Bit-depth Signals_ 

A PREPRINT 

Based on Table 1 and the additional details provided, it is evident that there are gaps to be filled in research on fault diagnosis in rotating machinery through data-driven approaches. Notably, the majority of the studies relied on proprietary data that were not made publicly available, rendering any comparative analysis limited to techniques impractical, as researchers can easily argue that any potentially superior performance achieved by others was accomplished under different conditions than those addressed in the original paper. In our study, we chose to utilize the MaFaulDa dataset, not only because it contains data relevant to the problem we aim to solve, but also because it is publicly available, allowing anyone interested to access the dataset, verify our results, and explore alternative approaches. 

There are very few studies addressing the problem of diagnosis in rotating machinery using sound exclusively as their sole source of information. Among these limited works, some employ multiple sound sources. This study, in addition to relying solely on sound as the source of information, utilizes a one-dimensional sound source. 

Another problematic point we found in other studies is that there doesn’t seem to be a concern regarding the sampling and quantization configurations of signals, as the evaluated works do not appear to have investigated whether the originally used configurations for dataset creation are in optimal conditions, without the possibility of utilizing something more parsimonious. Our work addressed this concern, demonstrating that a significantly more parsimonious approach is not only possible but also advisable, allowing for the development of solutions that require less computational power to achieve good results. Moreover, it provides insight into what the outcomes would be if different sampling and quantization configurations had been employed. 

Still regarding the sampling and quantization configurations, several studies merely utilized the original settings of the datasets, with sampling frequencies reaching up to 50 kHz, as is the case with MaFaulDa, and 24 bits. In contrast, we focused on working with even quite critical scenarios, such as 8 kHz and 8 bits, where we achieved some of the best results, which is part of one of our contributions in this work. 

It is important to remember another aspect pointed out by Lei et al. [3] in their discussion on challenges and the roadmap in IFD: On-line IFD. When working with online algorithms, it is very convenient to achieve the desired task with the least amount of information possible. 

Regarding the windowing of signals, we did not find any work that took into account the influence of window duration and overlap duration. This is quite relevant for the classification task. Subtle changes in the windowing configuration can produce substantial changes in the outcome of the diagnostic task. The fact that there are noticeable differences in performance from relatively minor changes highlights the need for this not to be overlooked; nevertheless, we still did not find anyone dedicated to specifically evaluating this. Fortunately, our work addressed this issue and conducted a thorough investigation into both the window duration and the overlap duration. 

The issue of preprocessing is another serious problem we noted, as some works did not provide information about what was done with the data during the preprocessing stage; or, what can be just as bad, provided incomplete or inaccurate information about what was done, leaving room for many doubts. There are also cases where what was done in one work may have been similar to what was done in another, but since different words were used to refer to certain techniques or approaches, there is no way to guarantee that what was done was necessarily the same. In the case of our work, on the other hand, we made a point of detailing each step taken and sought to use the same terminology typically employed in the field to avoid ambiguities. 

Regarding the features used, there is not always much clarity either. Some works barely use the term “features”, do not specify what types of features were utilized, and it can be challenging to ascertain the exact number of features employed. In contrast, our work clearly states what the features are and even specifies the position of each one in the feature vector. 

Many works were not clear regarding the use of any cross-validation approach. In some cases, it seems that no cross-validation was performed at all. Other studies employed a simple holdout method, separating only a percentage of the data into training and testing groups, which, while not inherently wrong, can be statistically questionable. Most studies that explicitly mentioned using some form of cross-validation utilized k-Fold, with k=10 being the dominant choice. However, it should be noted that there is no mention of whether stratified k-Fold was used; thus, it is safer to assume that it was not, which means that the folds may not have preserved the original class distributions. In contrast, our work employed a robust approach, specifically 10-fold stratified cross-validation. 

The variety of classes in the analyzed works is quite limited. In general, these studies considered only the main classes of the dataset. This approach means that the task is only fulfilled in relation to a fraction of what could be addressed. Some works have placed themselves in a slightly more complex situation by addressing the problem, even considering subclasses of fault types; however, despite solving a larger portion of the problems, it still addresses only a fraction. The portion of the problem that is not tackled by these studies is substantial, particularly concerning the intensity of the faults. In our work, we specifically addressed this issue by considering each different intensity of each fault, as well as 

10 

_IFD of Type and Severity in Low-Frequency, Low Bit-depth Signals_ 

A PREPRINT 

the class corresponding to the case where there is no problem at all, as entirely independent classes in the machine diagnostic task. This approach allowed us to frame the problem as one involving 42 classes, rather than just 10 or 6, as seen in other works. 

Many of the works employed traditional classifiers that are quite competent but can be costly to train. Others, on the other hand, may achieve good results but face more challenges regarding the explainability of what the classifier ultimately deemed most relevant for making the diagnosis. In our work, the classifier used is XGBoost, an ensemble based on decision trees, which allows us to pinpoint precisely which features had the most influence on the classification. 

Not all studies utilized balanced datasets. The MaFaulDa dataset, which has been used in several works and is the one we use in our study, is evidently unbalanced. Even acknowledging this, not all researchers appear to have taken special precautions to address this inconvenient characteristic of the dataset. In our work, however, we employed oversampling techniques, resulting in data augmentation that helped balance the data across all classes. We even present a performance comparison of each of the techniques employed. 

The following points summarize the key contributions of our study: 

- utilizes the publicly available MaFaulDa dataset. 

- focuses exclusively on a one-dimensional sound source. 

- demonstrates the feasibility of more parsimonious approaches, like 8 kHz, 8-bit. 

- categorizes each intensity of fault as an independent class, resulting in a total of 42 classes. 

- applies oversampling to enhance class balance. 

- our XGBoost ensemble classifier enhances interpretability by allowing visualization of feature importance. 

- presents good results even when using only a small portion of the MFCCs’ Deltas as features. 

Let us now examine in greater detail the data upon which this study is based. 

## **3 Dataset** 

In this work, we use the Machinery Fault Database (MaFaulDa) [20]. This dataset is constructed from a network of different sensors. In total, there are 8 different data sources, with only the last source corresponding to an audio signal collected with the aid of a single microphone. The MaFaulDa data is originally provided in Comma-Separated Values (CSV) files. Each column across a row corresponds to a source of information collected from a sensor or microphone. Each row represents a sample over time. There are a total of eight columns, but only the eighth column contains the sound signal, which is the sole information used in this work. Given that our study relies solely on the one-dimensional sound signal from MaFaulDa dataset, any additional information about the dataset that does not pertain to the sound signal will be disregarded. 

To gain a clearer understanding of potential distributions through alternative approaches with this dataset, it is useful to refer to Figure 2, which illustrates the data distribution when considering only the primary classes. In this approach, as adopted by other studies referenced in Section 2, only six total classes are represented. This method presents the most severe imbalance in sample count per class. 

Observe how the “normal” class comprises only 49 samples, while the smallest among the remaining classes, “horizontal_misalignment,” contains 197 samples, and the largest, “underhang_bearing,” includes 558 samples. From a binary perspective, in which only two classes — “healthy” and “faulty” — are considered, there would be 49 samples for “healthy” and 1902 samples for “faulty.” This corresponds to approximately 2.5% for “healthy” and 97.5% for “faulty.” 

In such a scenario, any approach that disregards such a pronounced imbalance would be highly inadvisable. 

Another approach, also employed by other studies mentioned in Section 2, is to consider that, beyond the main classes, the subclasses within two of the main classes could be treated as independent classes. In this case, instead of categorizing everything under “overhang_bearing” as a single class, the “overhang_bearing” class would be subdivided into three independent classes: “overhang_bearing_cage_fault,” “overhang_bearing_outer_race,” and “overhang_bearing_ball_fault.” A similar breakdown would apply to the “underhang_bearing” class and its subclasses. This approach would increase the number of classes from six to ten. 

In Figure 3, the full sample distribution across each of the 10 classes in this alternative approach can be observed. Notably, since this approach does not group all subclasses within “overhang_bearing” and “underhang_bearing” under their respective parent classes, but rather treats each subclass as an independent class, these two previously dominant 

11 

_IFD of Type and Severity in Low-Frequency, Low Bit-depth Signals_ 

A PREPRINT 

**Figure 2:** MaFaulDa samples distribution grouped by main classes. 

**==> picture [352 x 579] intentionally omitted <==**

**----- Start of picture text -----**<br>
600<br>500<br>400<br>300<br>200<br>100<br>0<br>class<br>Figure 3: MaFaulDa samples distribution grouped by subclasses.<br>350<br>300<br>250<br>200<br>150<br>100<br>50<br>0<br>class<br>normalhorizontal_misalignmentvertical_misalignment imbalanceoverhang_bearingunderhang_bearing<br>horizontal_misalignmentnormalvertical_misalignmentoverhang_bearing_cage_faultimbalanceoverhang_bearing_outer_raceoverhang_bearing_ball_faultunderhang_bearing_cage_faultunderhang_bearing_outer_raceunderhang_bearing_ball_fault<br># samples<br># samples<br>**----- End of picture text -----**<br>


12 

_IFD of Type and Severity in Low-Frequency, Low Bit-depth Signals_ 

A PREPRINT 

classes are no longer overwhelmingly represented in the dataset, as seen in the six-class case (see Figure 2). Nevertheless, there remains a noticeable imbalance, still disadvantaging the “normal” class. 

For the purposes of this study, it is important to emphasize that the diagnosis performed here will be based on both the type of fault and the intensity of the fault. Consequently, each case of varying intensity within each type of fault will be treated as an independent class. For instance, although they belong to the same general fault type, an underhang bearing ball fault will be considered distinct from an underhang bearing cage fault. Furthermore, regarding imbalance, cases with 10 g, 15 g, 20 g, 25 g, 30 g, and 35 g will be treated as separate and independent classes; thus, the 10 g case and the 15 g case are regarded as different classes. Each class has its own identification number, and it is crucial to note that this dataset is imbalanced, which means that different classes may contain varying numbers of samples. You can refer to Table 2 for detailed information about the classes and their respective sample distribution. 

It is relevant here to briefly clarify the 0 g cases within the “Cage Fault”, “Outer Race”, and “Ball Fault” subclasses of the “Overhang Bearing” and “Underhang Bearing” categories. A cursory review might mistakenly suggest that, since these values are zero, such samples should be classified under the “Normal” category; however, this would be incorrect. These bearing fault cases were generated using purposefully defective bearings, designed to present faults even though their diagnosis is more challenging. Thus, despite being 0 g cases, they indeed represent faulty classes. 

Table 3 presents the main specifications of the audio signal as originally found in MaFaulDa. The original sampling frequency of 50 Hz is atypical among audio signal processing works, but this is not an issue as it is even higher than the standards typically used in the field. This sampling frequency was adopted to conform to the sampling standards of other sensors in the original dataset, none of which deal with audio signals. 

**Figure 4:** MaFaulDa samples distribution. 

**==> picture [422 x 175] intentionally omitted <==**

**----- Start of picture text -----**<br>
50<br>40<br>30<br>20<br>10<br>0<br>0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41<br>class<br># samples<br>**----- End of picture text -----**<br>


We can visually examine the distribution of this data through Figure 4. Note that, although there is a certain balance among many of the classes, some exhibit severe imbalance. The classes corresponding to identification numbers 28 and 29 are those with the fewest samples. These are, respectively, the classes “overhang_bearing_ball_fault_20g” and “overhang_bearing_ball_fault_35g”; that is, they represent two scenarios within the same class and subclass, differing only in fault intensity. To give an idea of the extent of this imbalance: while most classes contain around 50 samples, class 28 has 25 samples, and class 29 has only 20 samples. 

With that, we can move on to preparing the data. 

## **4 Data Preprocessing** 

We already have the original data, but it still needs to be carefully analyzed, as adjustments may be necessary to make it suitable for use in model training. Therefore, it is necessary to prepare the data so that it is properly adjusted for training, and that is what we have done here, starting with the conversion of the signals, which had a distinct pattern from what we consider to be more suitable for this type of work. 

An important contribution of this work lies in examining the influence of various configurations and options. Rather than presuming an influence, or lack thereof, we opted to investigate experimentally. Consequently, instead of relying 

13 

_IFD of Type and Severity in Low-Frequency, Low Bit-depth Signals_ 

A PREPRINT 

**Table 2:** Distribution of measurement data across different classes, subclasses, and levels of severity. 

|**Class ID**<br>**Class**|**Subclass**<br>**Severity**<br>**Measurements**|
|---|---|
|0<br>Normal (no fault)|49|
|1<br>Horizontal Misalignment<br>2<br>3<br>4|-<br>0.50 mm<br>50<br>-<br>1.00 mm<br>49<br>-<br>1.50 mm<br>49<br>-<br>2.00 mm<br>49|
|5<br>Vertical Misalignment<br>6<br>7<br>8<br>9<br>10|-<br>0.51 mm<br>51<br>-<br>0.63 mm<br>50<br>-<br>1.27 mm<br>50<br>-<br>1.40 mm<br>50<br>-<br>1.78 mm<br>50<br>-<br>1.90 mm<br>50|
|11<br>Imbalance<br>12<br>13<br>14<br>15<br>16<br>17|-<br>6 g<br>49<br>-<br>10 g<br>48<br>-<br>15 g<br>48<br>-<br>20 g<br>49<br>-<br>25 g<br>47<br>-<br>30 g<br>47<br>-<br>35 g<br>45|
|18<br>Overhang Bearing<br>19<br>20<br>21<br>22<br>23<br>24<br>25<br>26<br>27<br>28<br>29|Cage Fault<br>0 g<br>49<br>6 g<br>49<br>20 g<br>49<br>35g<br>41|
||Outer Race<br>0 g<br>49<br>6 g<br>49<br>20 g<br>49<br>35g<br>41|
||Ball Fault<br>0 g<br>49<br>6 g<br>43<br>20 g<br>25<br>35 g<br>20|
|30<br>Underhang Bearing<br>31<br>32<br>33<br>34<br>35<br>36<br>37<br>38<br>39<br>40<br>41|Cage Fault<br>0 g<br>49<br>6 g<br>48<br>20 g<br>49<br>35g<br>42|
||Outer Race<br>0 g<br>49<br>6 g<br>49<br>20 g<br>49<br>35g<br>37|
||Ball Fault<br>0 g<br>50<br>6 g<br>49<br>20 g<br>49<br>35 g<br>38|



14 

_IFD of Type and Severity in Low-Frequency, Low Bit-depth Signals_ 

A PREPRINT 

**Table 3:** MaFaulDa audio signal specifications. 

|**Specifcation**|**Value**|
|---|---|
|Sampling Frequency|50 kHz|
|Bit-depth|24-bit|
|Encoding|raw (pure data from CSV fle)|
|Audio Channels|1 (8-th channel only)|
|Duration per File|5 seconds|
|Total Duration|_≈_2 hours and 42 minutes|
|Number of Files|1951 fles|



on a single possible configuration, we chose to test multiple configurations to determine which yields the most effective results. In this way, we illustrate some of the preprocessing approaches explored in this study. 

## **4.1 Audio signal conversion** 

Working with data in CSV format is not ideal; therefore, we generated a WAV file from the sound signal captured by the microphone, found in the eighth column of each CSV file. In this study, we explore various sampling and quantization configurations, which led us to create multiple versions with different combinations. 

**Table 4:** Sampling and quantization specifications. 

|**Specifcation**|**Value**|
|---|---|
|Sampling Frequency|{8, 16, 24, 48} kHz|
|Bit-depth|{8, 16, 24}-bit PCM|
|Encoding|Wav|



From Table 4, we can see the various options for sampling frequency, quantization bit depth, and encoding schemes applied in this study. 

## **4.2 Normalization and silence removing** 

At this point, the audio data had been converted into a much more convenient format, yet no assessment of their characteristics had been performed. We also aimed to investigate the impact that signal normalization could have on diagnostics. For this reason, we established a sweep range for this investigation, spanning from -12 dBFS to 0 dBFS. 

Given the large number of files, manually inspecting each one to identify any relatively long segments of silence would be impractical. Ignoring this issue, especially if a substantial number of such cases exist, could mean that a significant amount of samples produced during the feature extraction stage might originate from non-contributive, potentially detrimental instances for model training. To address this, we also chose to investigate the effect of removing silent intervals from each audio signal. For this purpose, we defined a range of possible thresholds for detecting silence, spanning from -48 dBFS to -15 dBFS. 

In both cases, we conducted an initial low-resolution sweep, meaning we adopted relatively large step sizes. We then refined the sweep resolution, focusing on regions where certain values had yielded more notable performance gains. 

## **4.3 Denoising** 

Although the MaFaulDa dataset was not specifically designed for denoising, the recordings made with the microphone were not conducted in an anechoic chamber. The microphone used is a Shure SM81, which is a highly sensitive condenser microphone widely recognized in the industry for capturing instruments. Consequently, there may be subtle background noises that, if addressed—even with low intensity during this enhancement phase—could yield improvements in the final diagnostic process while preserving the relevant information within the signals. 

15 

_IFD of Type and Severity in Low-Frequency, Low Bit-depth Signals_ 

A PREPRINT 

To perform this denoising, we chose to use Wiener filtering. More specifically, we utilized the implementation provided by the Scipy library. This implementation requires only two adjustable parameters as input arguments: the filter size and the power of the noise to be addressed. Regarding the noise power, we opted for the standard approach of the function, which calculates the power from the input signal itself. For the filter size, we decided to conduct a grid search between 1 and 35. 

## **4.4 Feature extraction** 

The entire process was conducted in Python programming language with the assistance of the Numpy, Scipy, Soundfile, Librosa, and Pandas libraries. However, it is important to note that the main libraries used for feature extraction itself are Numpy, Scipy, and Librosa; the others were used in a supporting role. 

Feature extraction starts with audio segmentation using the Hann window, given by: 

**==> picture [348 x 25] intentionally omitted <==**

Since we are still uncertain about which values of window duration and overlap duration would yield the best results for the diagnostic task, we conducted a grid search with window durations ranging from 300 ms to 600 ms and overlap durations from 0 ms to 200 ms. 

**Table 5:** Extracted features. 

|**Name**|**Domain**|**Components**|**Vector Position**|
|---|---|---|---|
|Mean|Time|1|[0]|
|Median|Time|1|[1]|
|Variance|Time|1|[2]|
|Standard Deviation|Time|1|[3]|
|Skewness|Time|1|[4]|
|Kurtosis|Time|1|[5]|
|Entropy|Time|1|[6]|
|Energy|Time|1|[7]|
|Power|Time|1|[8]|
|Minimum|Time|1|[9]|
|Maximum|Time|1|[10]|
|Peak to Peak distance|Time|1|[11]|
|Root Mean Square (RMS)|Time|1|[12]|
|Zero Crossing Rate (ZCR)|Time|1|[13]|
|Spectral Centroid|Frequency|1|[14]|
|Spectral Bandwidth|Frequency|1|[15]|
|Spectral Rolloff|Frequency|1|[16]|
|Spectral Flatness|Frequency|1|[17]|
|Spectral Contrast|Frequency|6|[18–23]|
|Mel-Frequency Cepstral Coeffcients (MFCC)|Mel-Frequency|40|[24–63]|
|Delta MFCC|Mel-Frequency|40|[64–103]|
|Delta2 MFCC|Mel-Frequency|40|[104–143]|
||**Total**|144||



Table 5 shows all the features extracted with the aim of composing the final dataset for the subsequent classification process. Among the presented features, there are several temporal features, spectral features, and Mel-frequency features. Some features have more than one component, such as spectral contrast and all the Mel-frequency features 

16 

_IFD of Type and Severity in Low-Frequency, Low Bit-depth Signals_ 

A PREPRINT 

used. Therefore, it is important to observe both the number of components and the position in the feature vector of each segment used in the feature extraction. 

At the end of the feature extraction process, some additional data were appended to the end of the feature vector for each sample in the data matrix. The following data were included: the original filename containing the signal from which the segment that produced the feature vector originated; the class label to which the sample belongs; the numerical class identifier; and the source type of the sample, which could be “original” or “generated”. With this, a CSV file containing all the extracted features, along with labels and other relevant data, was saved for use in subsequent processing steps. 

The information about the filename is crucial for ensuring the validity of the classification. Since the audio signals are segmented, a single audio file can give rise to multiple segments, and each segment will be used to extract features. 

If the dataset is prepared without a way to trace which audio file each sample originated from, it is possible that both the training and test groups could contain samples from the same original audio files. This would mean that part of the information used for training could be available during testing, significantly easing the testing process. 

Therefore, we also label each sample with the filename from which it originated. This allows for proper data partitioning before the training stage, ensuring that if any sample from a given audio file is allocated to the training group, all other samples from the same audio file are also allocated to the training group and none to the test group. 

From here, the data, which are now no longer audio signals but extracted features accompanied by their respective labels for each sample in the dataset, are ready to be used in the classification process. 

## **4.5 Data scaling** 

Before training the classifier, it is important to ensure that there is normalization across the entire dataset. This helps to avoid issues with features having different scales, can improve model performance, accelerate convergence during training, and prevent feature dominance. 

Given that this work involves 144 features, it would not be practical to display all of them here; therefore, we chose to illustrate the situation using only the time-domain data. Nonetheless, it is important to emphasize that, although only the time-based features are displayed here, all 144 features in the dataset will undergo the same scaling process. 

**Table 6:** Statistics of the temporal features as it is originally. 

|**Feature**<br>**Statistic**|**Mean**|**Median**|**Var.**|**Std.**|**Min**|**Max**|
|---|---|---|---|---|---|---|
|Mean|0_._0138|0_._0131|0_._0003|0_._0166|_−_0_._4349|0_._9944|
|Median|0_._0113|0_._0110|0_._0003|0_._0172|_−_0_._2578|1_._0000|
|Variance|0_._0020|0_._0018|0_._0000|0_._0068|0_._0000|0_._8060|
|Standard Deviation|0_._0431|0_._0428|0_._0001|0_._0112|0_._0068|0_._8978|
|Skewness|0_._7352|0_._8125|0_._3285|0_._5732|_−_15_._9717|2_._1781|
|Kurtosis|2_._9701|1_._3269|83_._8124|9_._1549|_−_1_._7828|271_._3044|
|Entropy|3_._0045|3_._0138|0_._0269|0_._1640|0_._1607|4_._6230|
|Energy|35_._3353|29_._3427|63 228_._8318|251_._4534|5_._9207|14 350_._3263|
|Power|0_._0025|0_._0020|0_._0003|0_._0175|0_._0004|0_._9966|
|Min|_−_0_._1277|_−_0_._0811|0_._0296|0_._1722|_−_1_._0000|_−_0_._0108|
|Max|0_._2246|0_._2084|0_._0071|0_._0841|0_._0580|1_._0000|
|Peak to Peak|0_._3523|0_._2911|0_._0581|0_._2411|0_._0757|2_._0000|
|Root Mean Square|0_._0457|0_._0451|0_._0004|0_._0192|0_._0203|0_._9983|
|Zero Crossing Rate|0_._1279|0_._1274|0_._0004|0_._0193|0_._0007|0_._1997|



From Table 6, even though we can only see a portion of what constitutes the entire feature matrix, we can note that the statistics of the features follow severely distinct patterns. While some values are enormous, others appear to be very small. In some cases, there seems to be almost no variability, while in others there appear to be large variations. In order to investigate which scaling method will yield the best result here, we will conduct experiments with various methods implemented in the Scikit-Learn library: MinMaxScaler, RobustScaler, and StandardScaler. 

17 

_IFD of Type and Severity in Low-Frequency, Low Bit-depth Signals_ 

A PREPRINT 

## **5 Data holdout** 

In the end, we will need to evaluate the performance of the model that will carry out the diagnosis, and this performance evaluation can only be conducted honestly if the samples used for assessing its performance are novel, meaning they are samples that have never been used, even partially, for this purpose. 

We are dealing with sounds from bearings that continue to rotate, between 700 RPM and 3600 RPM, which translates to approximately 11.7 Hz to 60.0 Hz, and the recordings have a duration of 5 seconds. This means that during a single recording, there have been approximately 58 to 300 rotations. 

Thus, there may be a reproduction of similar characteristics many times throughout the same trial. This is why, to avoid giving the classifier illicit advantages during evaluation, which would occur if it were trained with samples derived from a trial that also produced samples used in the final performance evaluation tests, we opted to perform a random separation of files between the training group and the testing group. 

We randomly selected 20% of all files from the original dataset to assign to the testing group. All files chosen for the testing group remained untouched throughout the entire processing, including the data augmentation process. Data augmentation was performed exclusively using samples from the training group, which comprises the remaining 80% of the files. 

We had not encountered any previous work that addressed this issue or emphasized the importance of performing such a separation in the data. Thus, there is no way to be certain that this step was carried out. And if it was done, it likely led to biased results. 

## **6 Data Augmentation** 

Oversampling will help address issues related to the imbalance observable from Figure 4. This will reduce the chances of the classifier suffering negatively from any type of bias that does not reflect the reality of the data. 

There are various data augmentation techniques that can be applied either to the original data or directly to the extracted features. We opted for an approach based on the already extracted features. 

Since this is also part of our investigation into what will yield the greatest performance gains for the diagnostic task, we will experiment with the techniques SMOTE, Borderline1 SMOTE, and Borderline2 SMOTE. The implementations from the ‘imbalanced-learn‘ library were utilized. 

Furthermore, each of these techniques allows for the selection of an oversampling strategy. We opted to conduct a grid search for this aspect as well. Initially, we performed a broad sweep with larger steps; then, we focused on areas that had yielded more promising results, performing a finer search with higher resolution to refine the tuning. 

## **7 Classification** 

Lei et al. [3] mentioned in Section 5.2 that, considering the big data revolution, it can be advantageous to use ensemble learning combined with some data resampling strategy. Mian et al. [38] reviewed 209 recent papers from 87 journals, analyzing 78 ensemble-based methods in IFD across 18 public datasets and 20+ systems. Notably, few studies used sound signals, with most focusing on vibration data. The limited use of ensemble learning with sound data suggests potential for exploring its advantages in fault diagnosis using one-dimensional acoustic signals. We opted for the classifier based on eXtreme Gradient Boosting (XGBoost) [39]. XGBoost is an ensemble method composed of a sequence of decision trees. Generally, it is a computationally lightweight approach and offers the advantage of interpretability, as it allows identification of the most significant features that contributed to the model’s performance. This interpretability can even be leveraged as a feature selection mechanism, aiding in dimensionality reduction. 

We used the implementation from the XGBoost library in Python, which allows customization of several hyperparameters, including the number of boosting rounds[*] , maximum tree depth, learning rate, the proportion of samples per iteration, and the feature fractions used at each tree and at each tree level. We conducted a grid search to optimize each of these parameters. 

To assess the model’s performance, we selected the metrics of accuracy, precision, recall, F1 Score, and F _β_ Score. 

**==> picture [146 x 22] intentionally omitted <==**

Also known as the number of trees in the processing chain. 

(2) 

18 

_IFD of Type and Severity in Low-Frequency, Low Bit-depth Signals_ 

A PREPRINT 

**==> picture [282 x 22] intentionally omitted <==**

**==> picture [277 x 22] intentionally omitted <==**

where TP stands for True Positives, TN stands for True Negatives, FP stands for False Positives, and FN stands for False Negatives; and, 

**==> picture [307 x 22] intentionally omitted <==**

**==> picture [325 x 25] intentionally omitted <==**

where _β_ is a constant that should be adjusted to choose whether the metric will favor precision or recall. The closer _β_ is to 0, the more precision is valued over recall; conversely, the larger the value, the more recall is valued over precision. When _β_ = 1, Equation (6) equals Equation (5), meaning that the F-1 Score is a particular case of F _β_ where the importance of precision equals the importance of recall. 

In many scenarios, it makes sense to equalize the importance of precision and recall, but this may not be the case for our work. Therefore, let’s understand the specifics of our particular situation. 

As we are dealing with mechanical fault diagnosis, we understand this is an area of potential risk, especially since, depending on the equipment and circumstances involved, there could be a risk of serious accidents or even fatality for those nearby. In such a scenario, if the classifier fails, it is preferable for the error to be a false positive, indicating a fault that did not actually occur, rather than a false negative, failing to indicate a fault that did occur. 

In other words, it is preferable for the error to be a false positive rather than a false negative; after all, a false positive error indicates a non-existent fault, which may result in a possibly manageable financial expense. On the other hand, a false negative error fails to notify an actual fault, posing risks to the safety of living beings and the entire environment where the equipment is located. 

In this case, the recall metric becomes more relevant than precision because it emphasizes the importance of false negatives over false positives. In Equation (6), we adopted _β_ = 2, which doubles the importance of recall compared to precision, so 

**==> picture [375 x 24] intentionally omitted <==**

We also chose to evaluate feature importance, enabling us to understand which features contributed most to the performance achieved in each simulation. Additionally, it is worth noting that numerous simulations were run to conduct a grid search across various scenarios. Initially, this search was performed with broad steps; however, after gaining a clearer understanding of the values that produced more favorable results, further searches were conducted with finer steps to refine the parameter selection. 

## **8 Workflow and simulation environment** 

Let’s understand a little better what the step-by-step process is. 

Figure 5 schematically shows a block diagram with all the main stages involved in the work, from the original dataset files to the classification. 

During the audio signal conversion stage, the signals originally recorded in CSV files were converted to WAV format, with different versions for sampling and quantization. Subsequently, the signals were normalized based on a given threshold and underwent silence removal according to a separate threshold. Then, Wiener filtering was applied to the signals. 

This was followed by the feature extraction stage for different windowing scenarios, considering both window duration and overlap duration, resulting in a feature matrix. This feature matrix was then normalized across each feature. Next, oversampling was applied to augment the data, addressing data imbalance. Finally, model training and evaluation were conducted, with the evaluation results used to fine-tune the parameters. 

19 

_IFD of Type and Severity in Low-Frequency, Low Bit-depth Signals_ 

A PREPRINT 

**Figure 5:** Representation of the complete process, from signal to classification. 

**==> picture [392 x 187] intentionally omitted <==**

**----- Start of picture text -----**<br>
Training<br>Data<br>Preprocessing (80%) Data Training<br>Augmenta�on Model<br>Audio signal conversion<br>Trained<br>Normaliza�on and silence removing Model<br>MaFaulDa Data<br>Dataset Denoising Holdout<br>Feature extrac�on Assessment<br>y<br>Data scaling Tes�ng x<br>Data<br>(20%)<br>Fine-tuning<br>**----- End of picture text -----**<br>


To conduct all the experiments in this study, considering that parsimony in computational resource consumption is essential, it would not make sense to use a supercomputer or a high-powered workstation. Therefore, we opted to use a personal desktop computer that was already accessible to us. 

**Table 7:** Main specifications of the computer used to train the model and perform simulations. 

|Component|Specifcations|
|---|---|
|CPU|Intel Core i5 8600K|
|GPU|NVidia GTX 1080 Ti 11 GB|
|RAM|32 GB (2 x 16 GB) DDR4 2400 MT/s|
|SSD|1 TB NVMe|
|S.O.|Ubuntu 22.04 LTS (WSL)|



Table 7 presents the main specifications of the desktop computer that was used to train the model and perform all simulations. 

## **9 Results** 

Various experiments were conducted. It is convenient to separate what each of them aims to evaluate. We can start with a standard scenario by setting fixed values for almost all possible parameters while varying only one of them. After going through each parameter, we will evaluate which value produced the best results, and from then on, that will be the new value for that parameter; we will then move on to the next parameter. 

Many rounds of fine-tuning adjustments were carried out. This makes it impractical to present each of the advances made throughout this work in detail. However, given the importance of understanding that the result achieved was the outcome of many small, consecutive improvements, we have created a diagram showcasing some of the key stages of enhancements produced through fine-tuning. This will allow us to observe some of the main milestones reached along the way. Refer to Figure 6. 

Next, we will present the results achieved through grid search variations for each of the main parameters. 

20 

_IFD of Type and Severity in Low-Frequency, Low Bit-depth Signals_ 

A PREPRINT 

**Figure 6:** Fine-tuning progress with its major milestones. 

**==> picture [310 x 568] intentionally omitted <==**

**----- Start of picture text -----**<br>
Scenario 1<br>ADASYN (per class): 200k<br>48 kHz, 24-bit<br>-6 dBFS norm, -48 dBFS silence<br>500 ms window, 20 ms overlap<br>All features<br>Robust Scaler<br>XGBoost: 1000 trees, up to 6 levels<br>Subsampling: 70%<br>ColSample_byTree: 80% Scenario 2<br>ColSample_byLevel: 80% 24-bit --> 8-bit<br>F-Beta Score: 99.93% 20 ms --> 0 ms overlap<br>1000 trees --> 200 trees<br>F-Beta Score: 100%<br>Scenario 3<br>48 kHz --> 8 kHz<br>200 trees --> 109 trees<br>F-Beta Score: 99.92% Scenario 4<br>Wiener Filter (size): 3<br>109 trees --> 60 trees<br>F-Beta Score: 99.49%<br>Scenario 5<br>-48 dBFS silence --> -20 dBFS silence<br>60 trees --> 50 trees<br>F-Beta Score: 99.18%<br>Scenario 6<br>50 trees --> 16 trees<br>6 levels --> 8 levels<br>Subsampling: 70% --> 80%<br>ColSample_byTree: 80% --> 70%<br>ColSample_byLevel: 80% --> 70%<br>Scenario 7<br>F-Beta Score: 99.48%<br>Wiener Filter (size): 3 --> 36<br>16 trees --> 12 trees<br>F-Beta Score: 99.02%<br>Scenario 8<br>ADASYN -->  Borderline-1 SMOTE<br>k-Neighbors: 5<br>m-Neighbors: 10<br>Scenario 9 200k samples/class --> 1k samples/class<br>F-Beta Score: 99.68%<br>Robust Scaler --> MinMax Scaler<br>12 trees --> 9 trees<br>F-Beta Score: 99.25%<br>Scenario 10<br>9 trees --> 8 trees<br>1k samples/class --> 960 samples/class<br>k-Neighbors: 5 --> 10<br>Scenario 11 m-Neighbors: 10 --> 2<br>8 trees --> 7 trees F-Beta Score: 99.29%<br>960 samples/class --> 616 samples/class<br>Subsample: 80% --> 90%<br>ColSample_byTree: 70% --> 20%<br>ColSample_byLevel: 70% --> 90%<br>F-Beta Score: 99.29%<br>Scenario 12<br>Wiener Filter (size): 36 --> 33<br>7 trees --> 6 trees<br>F-Beta Score: 99.52%<br>**----- End of picture text -----**<br>


21 

_IFD of Type and Severity in Low-Frequency, Low Bit-depth Signals_ 

A PREPRINT 

In some cases, minor adjustments will produce noticeable gains; in others, even significant variations may not lead to any perceptible change. Either way, each gain remains a favorable advancement, as long as it does not require parameter adjustments that severely increase computational demand for only a minimal performance improvement. 

Let’s begin with the sampling and quantization configurations. 

## **9.1 Influence of sampling and quantization** 

Let’s investigate how influential the sampling and quantization configurations are for the type of diagnosis performed in this work. 

**Figure 7:** Influence of sampling frequency and bit-depth. 

**==> picture [398 x 247] intentionally omitted <==**

**----- Start of picture text -----**<br>
1.00<br>0.99<br>0.98<br>0.97<br>Accuracy<br>Precision<br>0.96<br>Recall<br>F-1 Score<br>F-Beta<br>0.95<br>Sampling and Quantization Configurations<br>8 kHz, 8 bits8 kHz, 16 bits8 kHz, 24 bits16 kHz, 8 bits16 kHz, 16 bits16 kHz, 24 bits24 kHz, 8 bits24 kHz, 16 bits24 kHz, 24 bits48 kHz, 8 bits48 kHz, 16 bits48 kHz, 24 bits<br>F-Beta Score<br>**----- End of picture text -----**<br>


We can observe from Figure 7 that the configuration yielding the best result is 8 kHz and 8 bits. 

In this specific case, utilizing a substantially lower sampling frequency than the original, along with a significantly reduced bit quantization, yielded superior results. This configuration is advantageous as it facilitates embedding the solution in simpler devices while maintaining high performance in the diagnostic task. Furthermore, the parsimony of this setup leads to faster simulations, requires considerably less storage capacity, and does not rely on high throughput and bandwidth for data transfer, making it a highly convenient choice in this regard as well. 

## **9.2 Influence of signal normalization and silence removal** 

Now, we will use the sampling and quantization configuration that has produced the best results so far and conduct a grid search focusing on the input signal normalization threshold and the silence threshold. 

Let us first examine the influence of normalization. 

The normalization that delivered the best result, as shown in Figure 8, is the one with a threshold of -5.80 dBFS. In this case, it makes sense to take a pragmatic approach, as there are no significant differences in computational resource consumption; thus, we opted for -5.80 dBFS. 

Now we need to determine the best threshold for silence removal. Values very close to the normalization threshold will essentially classify almost everything as silence, leading to the removal of many crucial segments of information. Conversely, thresholds that are too low may result in almost nothing being considered silence, meaning that even segments where silence is distinctly present may not be removed. Therefore, this can be viewed as a sensitivity parameter to silence. 

22 

_IFD of Type and Severity in Low-Frequency, Low Bit-depth Signals_ 

A PREPRINT 

**Figure 8:** Influence of signal normalization threshold. 

**==> picture [398 x 247] intentionally omitted <==**

**----- Start of picture text -----**<br>
1.00<br>0.99<br>0.98<br>0.97<br>Accuracy<br>0.96 Precision<br>Recall<br>F-1<br>F-Beta<br>0.95<br>-7.0 -6.8 -6.6 -6.4 -6.2 -6.0 -5.8 -5.6 -5.4 -5.2 -5.0<br>Normalization Level (dBFS)<br>Metric Value<br>**----- End of picture text -----**<br>


**Figure 9:** Influence of silence removal threshold. 

**==> picture [398 x 247] intentionally omitted <==**

**----- Start of picture text -----**<br>
1.00<br>0.99<br>0.98<br>0.97<br>Accuracy<br>0.96 Precision<br>Recall<br>F-1 Score<br>F-Beta<br>0.95<br>-20.0 -19.5 -19.0 -18.5 -18.0<br>Silence Level (dBFS)<br>Score<br>**----- End of picture text -----**<br>


23 

_IFD of Type and Severity in Low-Frequency, Low Bit-depth Signals_ 

A PREPRINT 

The graph in Figure 9 shows that small variations in the threshold for what we consider silence to be removed can lead to significant changes in final performance. Despite the closeness of some results, it is clear that the threshold yielding the best outcome is -18 dBFS. 

## **9.3 Influence of denoising** 

Let us examine the influence of Wiener filter size on the final classification performance. 

**Figure 10:** Influence of Wiener filter size. 

**==> picture [469 x 232] intentionally omitted <==**

**----- Start of picture text -----**<br>
1.00<br>Accuracy<br>Precision<br>0.99 Recall<br>F-1 Score<br>0.98 F-Beta<br>0.97<br>0.96<br>0.95<br>0.94<br>0.93<br>0.92<br>1 3 5 7 9 11 13 15 17 19 21 23 25 27 29 31 33 35 37 39 41 43 45<br>Wiener Filter Size<br>Score<br>**----- End of picture text -----**<br>


The Wiener filter size is a parameter that significantly impacts classifier performance in the diagnostic task, as shown in Figure 10. The size that yielded the best result is 33. 

This is one of those cases where we observed a significant difference between different input values. Notice that when the filter sizes are 1, 3, or 5, none of the metrics even reach the 95% mark. However, starting from a size of 7, the filter begins to impact the classifier to the extent that some of the metrics nearly reach 99%. The standout value is 33, which delivered results exceeding 99.5%. 

## **9.4 Influence of window and overlap** 

Now we need to assess how impactful the window duration and overlap duration are on the classification performance. 

Clearly, as shown in Figure 11, the 500 ms window provided the best result. 

Here, we can clearly observe a significant difference between the maximum and minimum performance achieved by the different values of this parameter. It is noteworthy that this is simply a small adjustment in the window duration during the feature extraction stage. All other parameters were kept fixed; only the window duration was modified. Even with just this change, we were able to increase performance from below 88% to around 99.5%. Conversely, further increasing the window duration, such as moving from 500 ms to 600 ms, significantly harmed the final performance. Therefore, we settled on a duration of 500 ms. 

According to Figure 12, it is not beneficial to use overlap in the given scenario; after all, the best result was achieved without any overlap. 

The absence of overlap is somewhat unconventional. Overlap is generally applied to conduct a more thorough sweep along the signal, creating segments that might otherwise lack consideration of their latent information during segmentation. However, when dealing with signals whose latent information has already been effectively unwrapped and utilized, overlap may only add an unwanted element. In this case, it generates additional segments that do not bring 

24 

_IFD of Type and Severity in Low-Frequency, Low Bit-depth Signals_ 

A PREPRINT 

**Figure 11:** Influence of window length. 

**==> picture [398 x 247] intentionally omitted <==**

**----- Start of picture text -----**<br>
1.00<br>Accuracy<br>Precision<br>0.98 Recall<br>F-1 Score<br>F-Beta<br>0.96<br>0.94<br>0.92<br>0.90<br>0.88<br>0.86<br>400 500 600<br>Window Length (ms)<br>Score<br>**----- End of picture text -----**<br>


**Figure 12:** Influence of overlap length. 

**==> picture [398 x 247] intentionally omitted <==**

**----- Start of picture text -----**<br>
1.00<br>Accuracy<br>Precision<br>0.98 Recall<br>F1 Score<br>F-Beta<br>0.96<br>0.94<br>0.92<br>0.90<br>0.88<br>0.86<br>0 50 100 150<br>Overlap Length (ms)<br>Scores<br>**----- End of picture text -----**<br>


25 

_IFD of Type and Severity in Low-Frequency, Low Bit-depth Signals_ 

A PREPRINT 

any new information but rather produce redundant or even irrelevant segments, potentially leading to a drop in model performance. 

## **9.5 Influence of data scaling** 

Let’s analyze how our classifier performs with respect to the use of different data scaling techniques. 

**Figure 13:** Influence of data scaling. 

**==> picture [352 x 219] intentionally omitted <==**

**----- Start of picture text -----**<br>
1.0000<br>Accuracy<br>Precision<br>0.9975 Recall<br>F1 Score<br>0.9950 F-Beta<br>0.9925<br>0.9900<br>0.9875<br>0.9850<br>0.9825<br>0.9800<br>MinMax Standard Robust<br>Scaling Methods<br>Scores<br>**----- End of picture text -----**<br>


From Figure 13, it is evident that the scaling method that provided the best performance, exceeding 99.5%, was MinMax. This method scaled all the data of the same feature to the range [0, 1], where the lowest original values were very close to 0, and the highest values approached 1. 

## **9.6 Influence of data augmentation** 

First, let’s evaluate how each of the selected data augmentation methods performs in terms of their influence on diagnostic outcomes. 

From Figure 14, we can observe how the choice of data augmentation method impacts the final result obtained. Now, let’s examine how influential the number of samples generated by data augmentation is. 

As we can see from Figure 15, using 616 during this part of the simulations yielded the best result. 

As we can see, merely having more samples available is ineffective if the chosen method cannot generate samples that continue to contribute to accurately representing, in these generated samples, a faithful statistical depiction of the original dataset. If the data augmentation method simply produces additional samples without ensuring the quality of those samples, then—even with a higher sample count—the model’s performance will decline. 

## **9.7 Influence of number of boosting rounds** 

Let us now examine the influence of the number of boosting rounds, often interpreted as the number of trees involved in XGBoost’s sequential processing. This is one of the most commonly tuned parameters, and it’s important to remember that it can significantly impact the model, making it either much lighter or considerably heavier to train. Therefore, we might consider not selecting this parameter solely based on the highest performance achieved, as the best result may only slightly surpass another result obtained with a remarkably lower number of trees. This would substantially improve processing speed, making it a better choice for our balanced approach in this work. 

Refer to Figure 16. This chart provides insight into the number of trees required to nearly reach maximum classification performance. However, this study follows a different approach. Notice, for instance, that the result obtained with 6 trees is very close to the one with 12 trees. Although the 12-tree configuration does yield a better result, this minor 

26 

_IFD of Type and Severity in Low-Frequency, Low Bit-depth Signals_ 

A PREPRINT 

**Figure 14:** Influence of the data augmentation method. 

**==> picture [352 x 218] intentionally omitted <==**

**----- Start of picture text -----**<br>
1.00<br>Accuracy<br>Precision<br>0.98 Recall<br>F1 Score<br>F-Beta<br>0.96<br>0.94<br>0.92<br>0.90<br>0.88<br>0.86<br>SMOTE Random Borderline1 Borderline2<br>Augmentation Methods<br>Scores<br>**----- End of picture text -----**<br>


**Figure 15:** Influence of data augmentation samples. 

**==> picture [446 x 220] intentionally omitted <==**

**----- Start of picture text -----**<br>
1.000<br>Accuracy<br>Precision<br>Recall<br>0.995<br>F1 Score<br>F-Beta<br>0.990<br>0.985<br>0.980<br>0.975<br>0.970<br>600 601 602 603 604 605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620<br>Augmentation Samples<br>Scores<br>**----- End of picture text -----**<br>


27 

_IFD of Type and Severity in Low-Frequency, Low Bit-depth Signals_ 

A PREPRINT 

**Figure 16:** Influence of number of boosting rounds 

**==> picture [446 x 184] intentionally omitted <==**

**----- Start of picture text -----**<br>
1.00<br>Accuracy<br>Precision<br>0.95 Recall<br>F1 Score<br>0.90 F-Beta<br>0.85<br>0.80<br>0.75<br>0.70<br>0.65<br>0.60<br>1 2 3 4 5 6 7 8 9 10 11 12 13 14 15<br>Boosting Rounds<br>Scores<br>**----- End of picture text -----**<br>


performance improvement is not enough to justify the substantial increase in computational cost. The difference in model performance is marginal, making the 6-tree option more suitable in this context. 

## **9.8 Influence of max depth** 

Next, we will test the model’s performance by varying the maximum depth allowed for each tree in the XGBoost model. It is important to emphasize that this is a maximum limit, not an exact value, meaning some trees may not reach this depth. 

**Figure 17:** Influence of max depth. 

**==> picture [398 x 247] intentionally omitted <==**

**----- Start of picture text -----**<br>
1.0 Accuracy<br>Precision<br>Recall<br>F1 Score<br>0.8 F-Beta<br>0.6<br>0.4<br>0.2<br>0.0<br>1 2 3 4 5 6 7 8 9 10<br>Max Depth<br>Scores<br>**----- End of picture text -----**<br>


As we can observe from Figure 17, a maximum depth of 10 levels yields the best performance according to the adopted metrics. However, this is not an innocuous parameter regarding computational parsimony. This means that the greater the maximum number of levels that a tree can assume, the more expensive the proposed solution will be. Therefore, it 

28 

_IFD of Type and Severity in Low-Frequency, Low Bit-depth Signals_ 

A PREPRINT 

is desirable to find a balance here. In this case, the equilibrium point in question lies at a maximum depth of 8 levels, as it provides performance very close to that of the 10-level case, but with significantly lower computational cost. 

## **9.9 Influence of XGBoost internal sampling** 

XGBoost offers various sampling options concerning different stages of model processing. It is not always feasible to utilize all the samples available. There are several scenarios in which it is more advisable to use only a subset of the samples. The parameter in question allows for adjusting the percentage of samples that will be utilized. This percentage is randomly selected. In addition to making the model lighter, as it does not use part of the data, it can also enhance its flexibility. 

**Figure 18:** Influence of subsample. 

**==> picture [398 x 247] intentionally omitted <==**

**----- Start of picture text -----**<br>
1.0<br>Accuracy<br>Precision<br>Recall<br>0.9 F1 Score<br>F-Beta<br>0.8<br>0.7<br>0.6<br>0.5<br>0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9<br>Subsample<br>Scores<br>**----- End of picture text -----**<br>


The graph shown in Figure 18 demonstrates that using 90% of the available samples for training helped the model achieve superior performance in the diagnostic task. 

Another subsampling process within XGBoost involves selecting which features will be included in the pool of options at each new level of the same tree. 

Based on Figure 19, we can observe that using 90% of the features at each new level of the same tree provided the best performance. 

Finally, an additional sampling process performed by the XGBoost-based classifier is one in which, for each new tree introduced to the sequential processing chain, a percentage of the available features is selected for training. 

According to Figure 20, using only 20% of the features produced the best result obtained. 

## **9.10 Feature Importance** 

The features comprise one of the most crucial aspects of this study, as they drive nearly the entire learning process of the model and are also fundamental to its diagnostic capabilities. Let us now examine the importance of the top 20 features included in this work. Since 144 features were used, presenting all of them would be impractical. 

From Figure 21, we observe that the two primary features are purely statistical; specifically, they are the first central moments. Continuing with time-domain features, entropy and power appear relatively prominent compared to most others. The standout, however, was the MFCC, not so much on its own but particularly in relation to its Delta and Delta[2] components. Among these top features, there were no strictly frequency-domain-based features. Nearly all the main features in the graph displayed importance percentages that were quite close to each other. 

29 

_IFD of Type and Severity in Low-Frequency, Low Bit-depth Signals_ 

A PREPRINT 

**Figure 19:** Influence of column sample by level. 

**==> picture [398 x 247] intentionally omitted <==**

**----- Start of picture text -----**<br>
1.00<br>Accuracy<br>Precision<br>Recall<br>0.98 F1 Score<br>F-Beta<br>0.96<br>0.94<br>0.92<br>0.90<br>0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9<br>Colsample_bylevel<br>Scores<br>**----- End of picture text -----**<br>


**Figure 20:** Influence of column sample by tree. 

**==> picture [398 x 247] intentionally omitted <==**

**----- Start of picture text -----**<br>
1.00<br>Accuracy<br>Precision<br>Recall<br>0.99 F1 Score<br>F-Beta<br>0.98<br>0.97<br>0.96<br>0.95<br>0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9<br>Colsample_bytree<br>Scores<br>**----- End of picture text -----**<br>


30 

_IFD of Type and Severity in Low-Frequency, Low Bit-depth Signals_ 

A PREPRINT 

**Figure 21:** Feature importance. 

**==> picture [398 x 258] intentionally omitted <==**

**----- Start of picture text -----**<br>
mean 1.58%<br>variance 1.35%<br>delta_mfcc_28 1.12%<br>delta_mfcc_9 1.11%<br>delta_mfcc_39 1.06%<br>entropy 1.04%<br>delta_mfcc_8 1.03%<br>delta_mfcc_24 1.03%<br>delta_mfcc_6 1.02%<br>delta_mfcc_22 1.02%<br>delta_mfcc_1 1.01%<br>power 1.00%<br>delta_mfcc_4 0.99%<br>delta_mfcc_40 0.98%<br>delta_delta_mfcc_37 0.97%<br>delta_mfcc_30 0.96%<br>delta_mfcc_31 0.95%<br>delta_mfcc_25 0.94%<br>delta_delta_mfcc_34 0.94%<br>delta_mfcc_2 0.93%<br>0.00 0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00<br>Importance (%)<br>Feature<br>**----- End of picture text -----**<br>


## **9.11 MFCCs and greedy wrapper** 

After carefully evaluating the feature importance and recognizing the significant influence of MFCCs and their Deltas, we decided to conduct an additional experiment, training a model exclusively with these features. Since we had achieved impressive results using MFCC, we decided to conduct another experiment, this time using a greedy wrapper to select features one by one until the model no longer yielded performance gains, which resulted in even more remarkable outcomes. 

**Figure 22:** Diagnosis performance by feature selection using a greedy wrapper over MFCC and their first- and second-order Deltas. 

**==> picture [469 x 232] intentionally omitted <==**

**----- Start of picture text -----**<br>
1.0 Accuracy<br>Precision<br>Recall<br>F-1 Score<br>0.8 F-Beta<br>0.6<br>0.4<br>0.2<br>Selected Features<br>Score<br>delta_mfcc_3 delta_mfcc_13 delta_mfcc_28 delta_mfcc_17 delta_mfcc_33 delta_mfcc_6 delta_mfcc_36 delta_mfcc_7 delta_mfcc_22 delta_mfcc_8 delta_mfcc_14 delta_mfcc_2 delta_mfcc_5 delta_mfcc_19 delta_mfcc_15 delta_mfcc_30 delta_mfcc_37 delta_mfcc_4 delta_delta_mfcc_7 delta_mfcc_12 delta_mfcc_35 delta_mfcc_27 delta_mfcc_1 delta_delta_mfcc_30 delta_delta_mfcc_15 delta_mfcc_21 delta_delta_mfcc_17 delta_mfcc_23 delta_mfcc_18 delta_delta_mfcc_33 delta_mfcc_26 delta_delta_mfcc_5 delta_mfcc_29 delta_mfcc_11 delta_delta_mfcc_27 delta_mfcc_38 delta_mfcc_10 delta_mfcc_16 delta_delta_mfcc_12 delta_delta_mfcc_3 delta_delta_mfcc_11 delta_mfcc_34 delta_delta_mfcc_26 delta_delta_mfcc_13 delta_delta_mfcc_22 delta_delta_mfcc_38 mfcc_36 delta_delta_mfcc_29 delta_delta_mfcc_1 delta_mfcc_25<br>**----- End of picture text -----**<br>


31 

_IFD of Type and Severity in Low-Frequency, Low Bit-depth Signals_ 

A PREPRINT 

From Figure 22, it is evident that certain insights align with those drawn from Figure 21. The Deltas of the MFCCs play a significant role in classification, particularly the first-order Deltas. The wrapper was initially supplied with all the MFCC coefficients along with their first and second-order Deltas. The wrapper selected 50 features. Notably, only 1 of these was a MFCC, and 16 of the selected features were second-order Deltas. The remaining features chosen by the wrapper, as seen in Figure 22, consist exclusively of first-order Deltas. 

Another interesting point to observe is that if we split the progression into two parts of 25 iterations each, we notice that 22 of the first 25 selected features are first-order Deltas. In the second part, one selected feature is an MFCC, specifically one of the last calculated in this study—coefficient 36 out of 40. Among the remaining selections, 11 are first-order Deltas, and the other 13 are second-order Deltas. 

## **9.12 Best results and parameters** 

We will now present the best results achieved after implementing all adjustments, along with the parameters used to attain these results. 

**Table 8:** Best results achieved. 

|**Metric**|**All Features**<br>(144 Features)|**MFCCs and Deltas**<br>(120 Features)|**MFCCs and Deltas with Wrapper**<br>(50 Features)|
|---|---|---|---|
|Accuracy|0.9954|0.9783|0.9790|
|Precision|0.9954|0.9800|0.9920|
|Recall|0.9952|0.9767|0.9938|
|F1Score|0.9952|0.9774|0.9925|
|F_β_ Score|0.9952|0.9767|0.9932|



Table 8 presents the results for each of the performance evaluation metrics considered in this study, and it is important to emphasize that these results were obtained exclusively using the test group data. Note that there is a column for results obtained using all mentioned features, and another column for results obtained using only the MFCCs and their first- and second-order Deltas as features, and there is also another column in which only the MFCCs and their Deltas were used, but this time employing a greedy wrapper that performs feature selection one at a time and continues to add more features to the selected feature set for training the final model until including any additional feature ceases to enhance the model’s performance. 

After all these configuration and parameter adjustments throughout the entire process, we arrived at a final version that produced the best result achieved so far. 

Table 9 shows the final configurations and parameters that enabled us to achieve the best result using the MaFaulDa dataset. 

## **10 Conclusion** 

This study demonstrated that the task of diagnosing faults in rotating machinery, both in terms of fault type and fault severity, can be accomplished using sound as the sole source of information. Moreover, the research showed that it is feasible to perform this task even with sampling and quantization configurations typically considered quite limited, such as 8 kHz and 8-bit. 

Instead of simply employing a single grid search step with a few parameters varied at wide intervals, we chose to conduct multiple refinement stages, where, at each new stage, the step size was adjusted to be progressively smaller. This approach allowed us to identify values that yielded significant improvements, especially for the constrained scenarios addressed in this work. This resulted in an accuracy of 99.54% and an F _β_ score of 99.52% using all 144 features. 

Additionally, it is important to highlight that even when using only the MFCCs and their Deltas, completely ignoring the other features, this approach achieved performance very close to that obtained with all features. In this case, an accuracy of 97.83% and an F _β_ score of 97.67% were achieved using 120 features. Thus, with the assistance of the greedy wrapper, we achieved an accuracy of 97.90% and an F _β_ score of 99.32% using 50 features. 

32 

_IFD of Type and Severity in Low-Frequency, Low Bit-depth Signals_ 

A PREPRINT 

**Table 9:** Final version of the parameters and settings that provided the best result. 

|**Parameter**|**Value**|
|---|---|
|Sampling Frequency|8 kHz|
|Bit-depth|8-bit|
|Signal normalization threshold|-5.8 dBFS|
|Silence removal threshold|-18.0 dBFS|
|Wiener flter size|33|
|Window length|500 ms|
|Overlap length|0 ms|
|Scaling method|MinMax [0, 1]|
|Data augmentation - method|Borderline-1 SMOTE|
|Data augmentation - k_neighbors|10|
|Data augmentation - m_neighbors|2|
|Data augmentation - samples|616|
|XGBoost - boosting rounds (trees)|6|
|XGBoost - max depth|8|
|XGBoost - samples subsample|0.9|
|XGBoost - feature sample by level|0.9|
|XGBoost - feature sample by tree|0.2|



This indicates that, while other features still play an influential role in the model’s performance, MFCCs and their Deltas may be sufficient to achieve high performance in diagnosing faults in rotary machinery based on one-dimensional sound. 

## **References** 

- [1] D. Smith, _Reliability, Maintainability and Risk: Practical Methods for Engineers_ . Elsevier Science, 2021, ISBN: 9780323912624. [Online]. Available: `https://books.google.com.br/books?id=mBs-EAAAQBAJ` . 

- [2] N.-H. Kim, D. An, and J.-H. Choi, _Prognostics and Health Management of Engineering Systems_ , en. Cham: Springer International Publishing, 2017, ISBN: 978-3-319-44742-1. DOI: `10.1007/978-3-319-44742-1` . [Online]. Available: `http://link.springer.com/10.1007/978-3-319-44742-1` (visited on 07/12/2024). 

- [3] Y. Lei, B. Yang, X. Jiang, F. Jia, N. Li, and A. K. Nandi, “Applications of machine learning to machine fault diagnosis: A review and roadmap,” _Mechanical Systems and Signal Processing_ , vol. 138, p. 106 587, 2020, ISSN: 0888-3270. DOI: `https://doi.org/10.1016/j.ymssp.2019.106587` . [Online]. Available: `https://www.sciencedirect.com/science/article/pii/S0888327019308088` . 

- [4] S. Lee and P. White, “The enhancement of impulsive noise and vibration signals for fault detection in rotating and reciprocating machinery,” _Journal of Sound and Vibration_ , vol. 217, no. 3, pp. 485–505, 1998, ISSN: 0022-460X. DOI: `https://doi.org/10.1006/jsvi.1998.1767` . [Online]. Available: `https://www.sciencedirect. com/science/article/pii/S0022460X98917679` . 

- [5] K. SHIBATA, A. TAKAHASHI, and T. SHIRAI, “Fault diagnosis of rotating machinery through visualisation of sound signals,” _Mechanical Systems and Signal Processing_ , vol. 14, no. 2, pp. 229–241, 2000, ISSN: 0888-3270. DOI: `https://doi.org/10.1006/mssp.1999.1255` . [Online]. Available: `https://www.sciencedirect. com/science/article/pii/S0888327099912554` . 

- [6] J. Lin, “Feature extraction of machine sound using wavelet and its application in fault diagnosis,” _NDT & E International_ , vol. 34, no. 1, pp. 25–30, 2001, ISSN: 0963-8695. DOI: `https://doi.org/10.1016/S09638695(00)00025-6` . [Online]. Available: `https://www.sciencedirect.com/science/article/pii/ S0963869500000256` . 

- [7] U. Benko, J. Petrovˇciˇc, Ð. Juriˇci´c, J. Tavˇcar, J. Rejec, and A. Stefanovska, “Fault diagnosis of a vacuum cleaner motor by means of sound analysis,” _Journal of Sound and Vibration_ , vol. 276, no. 3, pp. 781–806, 2004, ISSN: 0022-460X. DOI: `https://doi.org/10.1016/j.jsv.2003.08.041` . [Online]. Available: `https://www.sciencedirect.com/science/article/pii/S0022460X03011428` . 

33 

_IFD of Type and Severity in Low-Frequency, Low Bit-depth Signals_ 

A PREPRINT 

- [8] U. Benko, J. Petrovic, D. Juricic, J. Tavcar, and J. Rejec, “An approach to fault diagnosis of vacuum cleaner motors based on sound analysis,” _Mechanical Systems and Signal Processing_ , vol. 19, no. 2, pp. 427–445, 2005, ISSN: 0888-3270. DOI: `https://doi.org/10.1016/j.ymssp.2003.09.004` . [Online]. Available: `https://www.sciencedirect.com/science/article/pii/S0888327003001146` . 

- [9] E. Olsson, P. Funk, and M. Bengtsson, “Fault diagnosis of industrial robots using acoustic signals and case-based reasoning,” in _Advances in Case-Based Reasoning_ , P. Funk and P. A. González Calero, Eds., Berlin, Heidelberg: Springer Berlin Heidelberg, 2004, pp. 686–701, ISBN: 978-3-540-28631-8. 

- [10] J.-D. Wu and C.-H. Liu, “An expert system for fault diagnosis in internal combustion engines using wavelet packet transform and neural network,” _Expert Systems with Applications_ , vol. 36, no. 3, Part 1, pp. 4278–4286, 2009, ISSN: 0957-4174. DOI: `https://doi.org/10.1016/j.eswa.2008.03.008` . [Online]. Available: `https://www.sciencedirect.com/science/article/pii/S0957417408001814` . 

- [11] M. Amarnath, V. Sugumaran, and H. Kumar, “Exploiting sound signals for fault diagnosis of bearings using decision tree,” _Measurement_ , vol. 46, no. 3, pp. 1250–1256, 2013, ISSN: 0263-2241. DOI: `https://doi. org/10.1016/j.measurement.2012.11.011` . [Online]. Available: `https://www.sciencedirect.com/ science/article/pii/S0263224112004320` . 

- [12] D. Pandya, S. Upadhyay, and S. Harsha, “Fault diagnosis of rolling element bearing with intrinsic mode function of acoustic emission data using apf-knn,” _Expert Systems with Applications_ , vol. 40, no. 10, pp. 4137–4145, 2013, ISSN: 0957-4174. DOI: `https://doi.org/10.1016/j.eswa.2013.01.033` . [Online]. Available: `https://www.sciencedirect.com/science/article/pii/S0957417413000468` . 

- [13] E. Germen, M. Ba¸saran, and M. Fidan, “Sound based induction motor fault diagnosis using kohonen selforganizing map,” _Mechanical Systems and Signal Processing_ , vol. 46, no. 1, pp. 45–58, 2014, ISSN: 0888-3270. DOI: `https : / / doi . org / 10 . 1016 / j . ymssp . 2013 . 12 . 002` . [Online]. Available: `https : / / www . sciencedirect.com/science/article/pii/S088832701300647X` . 

- [14] Y. Shatnawi and M. Al-khassaweneh, “Fault diagnosis in internal combustion engines using extension neural network,” _IEEE Transactions on Industrial Electronics_ , vol. 61, no. 3, pp. 1434–1443, 2014. DOI: `10.1109/ TIE.2013.2261033` . 

- [15] A. Moosavian, M. Khazaee, G. Najafi, M. Kettner, and R. Mamat, “Spark plug fault recognition based on sensor fusion and classifier combination using dempster–shafer evidence theory,” _Applied Acoustics_ , vol. 93, pp. 120–129, 2015, ISSN: 0003-682X. DOI: `https://doi.org/10.1016/j.apacoust.2015.01.008` . [Online]. Available: `https://www.sciencedirect.com/science/article/pii/S0003682X15000109` . 

- [16] S. Lu, X. Wang, Q. He, F. Liu, and Y. Liu, “Fault diagnosis of motor bearing with speed fluctuation via angular resampling of transient sound signals,” _Journal of Sound and Vibration_ , vol. 385, pp. 16–32, 2016, ISSN: 0022-460X. DOI: `https://doi.org/10.1016/j.jsv.2016.09.012` . [Online]. Available: `https: //www.sciencedirect.com/science/article/pii/S0022460X16304771` . 

- [17] C. K. Madhusudana, H. Kumar, and S. Narendranath, “Face milling tool condition monitoring using sound signal,” en, _International Journal of System Assurance Engineering and Management_ , vol. 8, no. S2, pp. 1643– 1653, Nov. 2017, ISSN: 0975-6809, 0976-4348. DOI: `10.1007/s13198-017-0637-1` . [Online]. Available: `http://link.springer.com/10.1007/s13198-017-0637-1` (visited on 10/09/2024). 

- [18] C. Madhusudana, H. Kumar, and S. Narendranath, “Fault diagnosis of face milling tool using decision tree and sound signal,” _Materials Today: Proceedings_ , vol. 5, no. 5, Part 2, pp. 12 035–12 044, 2018, International Conference on Materials Manufacturing and Modelling, ICMMM - 2017, 9 - 11, March 2017, ISSN: 22147853. DOI: `https://doi.org/10.1016/j.matpr.2018.02.178` . [Online]. Available: `https://www. sciencedirect.com/science/article/pii/S221478531830381X` . 

- [19] K. A. Loparo, _CWRU bearing dataset_ . [Online]. Available: `https : / / engineering . case . edu / bearingdatacenter` . 

- [20] F. M. L. Ribeiro, _Mafaulda-machinery fault database_ , 2016. [Online]. Available: `https://www02.smt.ufrj. br/~offshore/mfs/page_01.html#SEC2` . 

- [21] F. Ribeiro, M. Marins, S. Netto, and E. Silva, “Rotating machinery fault diagnosis using similarity-based models,” en, in _Anais de XXXV Simpósio Brasileiro de Telecomunicações e Processamento de Sinais_ , Sociedade Brasileira de Telecomunicações, 2017. DOI: `10.14209/sbrt.2017.133` . [Online]. Available: `http://biblioteca. sbrt.org.br/articles/601` (visited on 10/06/2024). 

- [22] M. A. Marins, F. M. Ribeiro, S. L. Netto, and E. A. Da Silva, “Improved similarity-based modeling for the classification of rotating-machine failures,” en, _Journal of the Franklin Institute_ , vol. 355, no. 4, pp. 1913–1930, Mar. 2018, ISSN: 00160032. DOI: `10.1016/j.jfranklin.2017.07.038` . [Online]. Available: `https: //linkinghub.elsevier.com/retrieve/pii/S0016003217303678` (visited on 07/22/2024). 

34 

_IFD of Type and Severity in Low-Frequency, Low Bit-depth Signals_ 

A PREPRINT 

- [23] D. S. C. Rocha, “Aprendizado de máquina aplicado ao reconhecimento automático de falhas em máquinas rotativas,” Master’s Thesis, Universidade Federal de Minas Gerais, 2018. [Online]. Available: `http://hdl. handle.net/1843/BUBD-B4PP45` . 

- [24] Y. Wang, N. Liu, H. Guo, and X. Wang, “An engine-fault-diagnosis system based on sound intensity analysis and wavelet packet pre-processing neural network,” _Engineering Applications of Artificial Intelligence_ , vol. 94, p. 103 765, 2020, ISSN: 0952-1976. DOI: `https://doi.org/10.1016/j.engappai.2020.103765` . [Online]. Available: `https://www.sciencedirect.com/science/article/pii/S0952197620301688` . 

- [25] Y. Sun, Y. Cao, G. Xie, and T. Wen, “Sound based fault diagnosis for rpms based on multi-scale fractional permutation entropy and two-scale algorithm,” _IEEE Transactions on Vehicular Technology_ , vol. 70, no. 11, pp. 11 184–11 192, 2021. DOI: `10.1109/TVT.2021.3090419` . 

- [26] A. Altinors, F. Yol, and O. Yaman, “A sound based method for fault detection with statistical feature extraction in uav motors,” _Applied Acoustics_ , vol. 183, p. 108 325, 2021, ISSN: 0003-682X. DOI: `https://doi.org/ 10.1016/j.apacoust.2021.108325` . [Online]. Available: `https://www.sciencedirect.com/science/ article/pii/S0003682X21004199` . 

- [27] B. Połok and P. Bilski, “Intelligent diagnostic system for the rachet mechanism faults detection using acoustic analysis,” _Measurement_ , vol. 183, p. 109 637, 2021, ISSN: 0263-2241. DOI: `https://doi.org/10.1016/ j.measurement.2021.109637` . [Online]. Available: `https://www.sciencedirect.com/science/ article/pii/S0263224121006084` . 

- [28] Y. Cao, Y. Sun, G. Xie, and P. Li, “A sound-based fault diagnosis method for railway point machines based on two-stage feature selection strategy and ensemble classifier,” _IEEE Transactions on Intelligent Transportation Systems_ , vol. 23, no. 8, pp. 12 074–12 083, 2022. DOI: `10.1109/TITS.2021.3109632` . 

- [29] A. Suman, C. Kumar, and P. Suman, “Early detection of mechanical malfunctions in vehicles using sound signal processing,” _Applied Acoustics_ , vol. 188, p. 108 578, 2022, ISSN: 0003-682X. DOI: `https://doi.org/10. 1016/j.apacoust.2021.108578` . [Online]. Available: `https://www.sciencedirect.com/science/ article/pii/S0003682X21006721` . 

- [30] Y. Zhu, G. Li, S. Tang, R. Wang, H. Su, and C. Wang, “Acoustic signal-based fault detection of hydraulic piston pump using a particle swarm optimization enhancement cnn,” _Applied Acoustics_ , vol. 192, p. 108 718, 2022, ISSN: 0003-682X. DOI: `https://doi.org/10.1016/j.apacoust.2022.108718` . [Online]. Available: `https://www.sciencedirect.com/science/article/pii/S0003682X22000925` . 

- [31] J. Pacheco-Chérrez, J. A. Fortoul-Díaz, F. Cortés-Santacruz, L. María Aloso-Valerdi, and D. I. Ibarra-Zarate, “Bearing fault detection with vibration and acoustic signals: Comparison among different machine leaning classification methods,” _Engineering Failure Analysis_ , vol. 139, p. 106 515, 2022, ISSN: 1350-6307. DOI: `https://doi.org/10.1016/j.engfailanal.2022.106515` . [Online]. Available: `https://www. sciencedirect.com/science/article/pii/S1350630722004897` . 

- [32] Y. Sha, J. Faber, S. Gou, _et al._ , “An acoustic signal cavitation detection framework based on xgboost with adaptive selection feature engineering,” _Measurement_ , vol. 192, p. 110 897, 2022, ISSN: 0263-2241. DOI: `https://doi.org/10.1016/j.measurement.2022.110897` . [Online]. Available: `https://www. sciencedirect.com/science/article/pii/S0263224122001798` . 

- [33] Y. E. Karabacak, N. Gürsel Özmen, and L. Gümü¸sel, “Intelligent worm gearbox fault diagnosis under various working conditions using vibration, sound and thermal features,” _Applied Acoustics_ , vol. 186, p. 108 463, 2022, ISSN: 0003-682X. DOI: `https://doi.org/10.1016/j.apacoust.2021.108463` . [Online]. Available: `https://www.sciencedirect.com/science/article/pii/S0003682X21005570` . 

- [34] T. Mian, A. Choudhary, and S. Fatima, “An efficient diagnosis approach for bearing faults using sound quality metrics,” _Applied Acoustics_ , vol. 195, p. 108 839, 2022, ISSN: 0003-682X. DOI: `https://doi.org/10.1016/ j.apacoust.2022.108839` . [Online]. Available: `https://www.sciencedirect.com/science/article/ pii/S0003682X22002134` . 

- [35] O. Das and D. Bagci Das, “Smart machine fault diagnostics based on fault specified discrete wavelet transform,” _Journal of the Brazilian Society of Mechanical Sciences and Engineering_ , vol. 45, no. 1, p. 55, Jan. 2023. 

- [36] Y. Yue, H. Wang, and S. Zhang, “Mel frequency mapping for intelligent diagnosis of rolling element bearings across different working conditions,” _Applied Acoustics_ , vol. 220, p. 109 944, 2024, ISSN: 0003-682X. DOI: `https://doi.org/10.1016/j.apacoust.2024.109944` . [Online]. Available: `https://www. sciencedirect.com/science/article/pii/S0003682X24000951` . 

- [37] R. K. Mishra, A. Choudhary, S. Fatima, A. Mohanty, and B. Panigrahi, “A generalized method for diagnosing multi-faults in rotating machines using imbalance datasets of different sensor modalities,” _Engineering Applications of Artificial Intelligence_ , vol. 132, p. 107 973, 2024, ISSN: 0952-1976. DOI: `https://doi.org/10.1016/ j.engappai.2024.107973` . [Online]. Available: `https://www.sciencedirect.com/science/article/ pii/S0952197624001313` . 

35 

_IFD of Type and Severity in Low-Frequency, Low Bit-depth Signals_ 

A PREPRINT 

- [38] Z. Mian, X. Deng, X. Dong, _et al._ , “A literature review of fault diagnosis based on ensemble learning,” _Engineering Applications of Artificial Intelligence_ , vol. 127, p. 107 357, 2024, ISSN: 0952-1976. DOI: `https: //doi.org/10.1016/j.engappai.2023.107357` . [Online]. Available: `https://www.sciencedirect. com/science/article/pii/S0952197623015415` . 

- [39] T. Chen and C. Guestrin, “Xgboost: A scalable tree boosting system,” in _Proceedings of the 22nd acm sigkdd international conference on knowledge discovery and data mining_ , 2016, pp. 785–794. 

36 

