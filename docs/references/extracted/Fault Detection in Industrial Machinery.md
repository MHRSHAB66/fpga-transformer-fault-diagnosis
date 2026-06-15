DEGREE PROJECT ee)or 

## Fault Detection in Industrial Machinery 

## Athanasios Koutoulas 

**Computer Science and Engineering, master's level (60 credits)** 2023-2024 

Luleå University of Technology Department of Computer Science, Electrical and Space Engineering 

## **Abstract** 

Technological advancements have increased the complexity of industrial machines. Simultaneously, global competition and economic factors have intensified the demands and harshness of industrial environments. These challenges elevate the risks of machinery failure, which can result in significant financial costs, production interruptions, safety hazards, and environmental accidents with far-reaching consequences. Therefore, timely detection of machinery faults is crucial. This study investigates the suitability of deep learning solutions for fault detection in industrial machinery. Specifically, nine fault detection models are implemented using various combinations of deep learning technologies, such as recurrent and convolutional networks, attention mechanisms, and transformers, to detect fault states in a rotating machine and an industrial gearbox. The models are trained in a supervised manner, and their performance is tested in two classification tasks. The evaluation shows that the deep learning models successfully captured the fault patterns of the rotating machine and the industrial gearbox, with eight out of nine models achieving accuracy scores above 97% in both classification tasks. Furthermore, the comparison of the explored solutions reveals that the attention mechanism and transformers are particularly effective for the domain. Finally, to gain a clearer understanding of the effectiveness of deep learning solutions in detecting machinery faults, the obtained results are compared with those from previous studies using traditional machine learning techniques on the same datasets. The comparison indicates that deep learning models achieve higher fault detection rates with less elaborate preprocessing techniques. 

**Keywords:** Industrial Fault Detection, Industrial Machinery, Machinery Fault Detection, Supervised Fault Detection, Machine Learning, Deep Learning. 

i 

## **Acknowledgments** 

First and foremost, I would like to thank my supervisors Dr. Muhammad Hanif (2022-2023) and Dr. Abdolrasoul Habibipour (2023-2024) for giving me the opportunity to work on this interesting topic under their supervision. Without their guidance, consistent support and invaluable advice, this work could not have been carried out. Moreover, I would like to express my sincere gratitude to my teachers at Luleå University of Technology for providing me with the necessary knowledge to accomplish this project. Finally, I am deeply thankful to my wife Joyce for her immense support throughout the duration of my studies. 

Athanasios Koutoulas Antwerp, Belgium September 2024 

ii 

## **Table of Contents** 

**Abbreviations** ........................................................................................................................ **1 1. Introduction** ...................................................................................................................... **2** 1.1 Fundamentals .................................................................................................................... 2 1.2 Background ....................................................................................................................... 5 1.3 Motivations ....................................................................................................................... 8 1.4 Problem Definition ......................................................................................................... 10 1.5 Objectives ....................................................................................................................... 11 1.6 Ethics .............................................................................................................................. 11 1.7 Sustainability .................................................................................................................. 11 1.8 Use of AI Tools .............................................................................................................. 12 1.9 Thesis Structure .............................................................................................................. 12 **2. Related Work** ................................................................................................................. **13** 2.1 Machine Learning Approaches ....................................................................................... 13 2.1.1 MaFaulDa Dataset ................................................................................................... 13 2.1.2 CSU Dataset ............................................................................................................. 15 2.2 Deep Learning Approaches ............................................................................................ 15 **3. Theoretical Underpinning** ......................................................................................... **21** 3.1 Fault Detection ................................................................................................................ 21 3.1.1 Fault Detection Methods .......................................................................................... 21 3.1.2 Data-Driven Condition Monitoring ......................................................................... 22 3.1.3 Early Fault Detection ............................................................................................... 23 3.2 Time Series ..................................................................................................................... 24 3.2.1 Feature Extraction .................................................................................................... 26 3.3 Artificial Neural Networks ............................................................................................. 27 3.3.1 Activation Functions ................................................................................................ 29 3.3.2 Long-Term Dependencies Challenge ...................................................................... 31 3.4 Deep Learning Networks ................................................................................................ 31 3.4.1 Recurrent Neural Network ....................................................................................... 31 3.4.2 Long Short-Term Memory Network ........................................................................ 32 3.4.3 Gated Recurrent Unit Network ................................................................................ 34 3.4.4 Convolutional Neural Network ................................................................................ 35 3.4.5 Attention Mechanism ............................................................................................... 37 

iii 

3.4.6 Transformer ............................................................................................................. 39 3.5 Performance Evaluation Metrics .................................................................................... 42 3.5.1 Confusion Matrix ..................................................................................................... 42 3.5.2 Accuracy .................................................................................................................. 42 3.5.3 Precision ................................................................................................................... 43 3.5.4 Recall ....................................................................................................................... 43 3.5.5 F-Score ..................................................................................................................... 43 3.5.6 ROC Curve .............................................................................................................. 43 **4. Implementation** ............................................................................................................. **45** 4.1 Pipeline ........................................................................................................................... 45 4.2 Experimental Environment ............................................................................................. 45 4.3 Datasets ........................................................................................................................... 45 4.3.1 MaFaulDa Dataset ................................................................................................... 46 4.3.2 CSU Dataset ............................................................................................................. 48 4.3.3 Exploration and Preprocessing ................................................................................ 50 4.4 Inference Tasks ............................................................................................................... 55 4.5 Model Architectures ....................................................................................................... 55 4.5.1 Simple RNN ............................................................................................................. 55 4.5.2 LSTM ....................................................................................................................... 56 4.5.3 GRU ......................................................................................................................... 57 4.5.4 CNN ......................................................................................................................... 58 4.5.5 CNN-LSTM ............................................................................................................. 59 4.5.6 CNN-GRU ............................................................................................................... 61 4.5.7 LSTM-Attention ...................................................................................................... 62 4.5.8 GRU-Attention ......................................................................................................... 64 4.5.9 Transformer ............................................................................................................. 65 4.6 Learning Process ............................................................................................................. 67 **5. Evaluation** ....................................................................................................................... **70** 5.1 Rotating Machine Fault Detection .................................................................................. 70 5.2 Industrial Gearbox Fault Detection ................................................................................ 83 **6. Discussion** ........................................................................................................................ **97** 6.1 Classification Tasks ........................................................................................................ 97 6.2 Reflection on the Research Questions ............................................................................ 99 6.2.1 Research Question 1 ................................................................................................ 99 6.2.2 Research Question 2 .............................................................................................. 100 

iv 

6.2.3 Research Question 3 .............................................................................................. 103 **7. Conclusion and Future Work** ................................................................................ **106** 7.1 Conclusion ................................................................................................................... 106 7.2. Contribution ............................................................................................................... 108 7.3. Limitations and Future Work ..................................................................................... 109 **8. References** ..................................................................................................................... **111** 

v 

## **Abbreviations** 

**I4.0** Industry 4.0 **IoT** Internet of Things **IIoT** Industrial Internet of Things **AI** Artificial Intelligence **ML** Machine Learning **DL** Deep Learning **ANN** Artificial Neural Network **DNN** Deep Neural Network **CNN** Convolutional Neural Network **RNN** Recurrent Neural Network **LSTM** Long Short-Term Memory **GRU** Gated Recurrent Unit **MLP** Multi-Layer Perceptron **SVM** Support Vector Machine **SMOTE** Synthetic Minority Oversampling Technique **CRISP-DM** Cross Industry Standard Process for Data Mining **FFT** Fast Fourier Transform **ROC** Receiver Operating Characteristic **AUC** Area under the Curve **MAFAULDA** Machinery Fault Database (dataset) **CSU** China Southeast University (dataset) 

1 

## **1. Introduction** 

## **1.1Fundamentals** 

In the modern era, technology plays a significant role in our everyday lives, but the increasingly connected culture we live in has an impact on the industrial world as well. Currently, we stand on the brink of a transformation in the way we produce products, thanks to the digitization of manufacturing. Experts refer to the ongoing transformation in the industrial world as its fourth revolution, known as Industry 4.0 (I4.0). What mainly concerns I4.0 is to create "smart factories" that are highly automated, connected, and data-driven, using advanced technologies such as the Internet of Things (IoT), cloud computing, and big data analytics [1]. These technologies facilitate dynamic communication between digital instruments, machines, and humans, thereby enhancing and allowing for the monitoring of productivity [2]. 

**Figure 1.1: The building blocks of I4.0.** 

When computers were introduced to manufacturing processes during the third industrial revolution (I3.0), it was riotous due to inclusion of an entirely new technology. As Industry 4.0 unfolds, the associated technologies will facilitate the digitalization of industrial equipment and enable machines to connect and dynamically exchange information. Computers as part of this process, have the ability to make decisions without human involvement, making the concept of smart factory a reality [3]. Smart factories aim to use information from various processes to optimize throughput, increase customer satisfaction, and reduce costs. 

On the other hand, the industrial transformation associated with smart manufacturing and the consolidation of I4.0 creates numerous challenges for organizations, technologies, and entire 

2 

societies. One of these challenges is the complexity of manufacturing systems that grew along with each industrial revolution [4]. 

**Figure 1.2: Increasing system complexity over the industrial revolutions [4].** 

Smart factories integrate intelligent systems and machines of high functional complexity driven by recent technological advancements. In many cases these machines operate in inhospitable and intensive environments led by global competition, multilevel business targets, and other economic factors. Moreover, diversity in product portfolios along with the increasing demand for delivering complex features to products, require integration of different types of machines in a single process which can impose several challenges [5]. 

This situation creates uncertainty and volatility for industrial machines. High workload and harsh environments with factors such as humidity, temperature, vibrations, make machines more prone to failures. Furthermore, as functional complexity increases, so does the likelihood of failures and the complexity of the underlying fault patterns. [6]. To remain competitive, manufacturers need to invest in effective methods that reduce machinery malfunctions and failures. 

Detecting fault in incipient stages can increase process reliability and reduce the risks of potential failures. More specifically, early fault detection in industrial equipment can drastically reduce unexpected system failures and unnecessary costs, it can contribute to cost-effective maintenance plans, and enhance safety for the operators. At this point, it is important to elucidate the definitions of fault and failure for better clarity. 

3 

According to [7], an industrial machinery fault can be defined as the occurrence of an unpermitted event that deviates a machine from its normal state or desired behavior, while failure can be defined as the complete interruption of a machine's ability to execute a demanded function, attributed to the occurrence of one or more faults. Additionally, it is important to distinguish machinery failure from malfunction, which can be defined as a sporadic interruption of a machine’s normal process due to faults that can eventually lead to failure. 

**Figure 1.3: Evolution of fault occurrence in a machine’s operating processes.** 

The technologies of I4.0 have the potential to detect and identify developing faults in a machine’s operating process and predict their progression towards failure. The Industrial IoT (IIoT) is an ecosystem comprised of sensors and digital tools, able to collect metrics from various machinery components and exchange information over networks [8]. Such information corresponds to temperatures, vibrations, acceleration speeds, power consumption values etc. 

As the complexity of smart machines increases, the volume of generated information grows along. I4.0 integrates the Cloud as a storage solution for the massive amount of information exchanged between IIoT devices [8, 9]. Cloud computing supports the growth of data sharing within smart industrial environments by providing scalable storage capabilities, increased computational power, and improved data accessibility and integrity. 

Cloud computing and Big Data Analytics form a symbiotic relationship, enabling combined capabilities for real-time processing and scalable data analysis. I4.0 accommodates the concept of Big Data Analytics to interpret information of interest from various industrial processes and systems by utilizing a wide range of techniques [9] including statistical models, algorithms, and Machine Learning (ML). 

ML techniques can play a key role in the concept of smart factories since they can leverage the available information to uncover hidden fault patterns, correlations and trends in the data, associated with machinery failure. Finally, the end goal of Big Data Analytics in industrial processes is to strengthen the performance of manufacturing facilities by increasing their reliability and at the same time enhance data-driven decision-making [9]. 

4 

## **1.2Background** 

In recent years, the field of fault detection in industrial machines has seen a surge in interest thanks to the remarkable advancements of ML and the widespread availability of I4.0 technologies. Consequently, ML has become an increasingly attractive topic for studies focusing on the field. More specifically, several studies have approached the problem with traditional ML techniques, and more recently with Deep Learning (DL) techniques. The following graph demonstrates a taxonomy of some traditional ML and DL solutions used in previous research in the domain [10]. 

**Figure 1.4: A taxonomy of traditional ML and DL techniques, used in domain of fault detection in industrial machines [10].** 

ML models can process data generated by industrial machines to detect potential faults at an early stage, before they evolve into failures [10, 11]. They do this by uncovering hidden patterns, correlations, and anomalies in data that may not be easily discernible by humans. This capability is especially valuable for identifying subtle or infrequent faults that may go unnoticed by traditional monitoring methods. Furthermore, ML models can be trained on historical data to improve their accuracy in identifying faults over time [12]. Additionally, they can be part of an automated decision-making process without human involvement, and they can be customized to meet specific needs of different industrial machines and processes. 

On the other hand, traditional ML algorithms (often referred to as shallow learning), can suffer some limitations and assumptions when it comes to detecting faults in industrial machines. The integration of IIoT in smart factories often results in massive volumes of data exchanged between physical and digital systems. High data dimensionality arising from multiple sensors, can also affect their performance [10, 11], while after some training iterations their performance can become stale [12]. 

5 

Additionally, some traditional ML algorithms assume that each data point in a sample is independent of the others [13]. This assumption may not hold true for time series data, as there may be temporal dependencies and shared patterns across multiple sequences. More specifically, the data collected from sensors and digital instruments, such as sound and vibration signals, have a temporal order. For capturing fault patterns deeply, it is important to consider the order of observations over a period of time, and their shared patterns with past observations. Therefore, learning from time series is more complicated since it requires models to find the correlations between time, parallel and past observations [14]. 

Another limitation of traditional ML algorithms is that in many cases they require manual feature extraction for generating the most informative features and increase model performance [10, 11, 12]. This is because they usually involve a limited number of trainable parameters, and they cannot automatically extract features of high relevance from the input data. Thus, a lot of effort is often put into manual feature extraction. 

The limitations of traditional ML models motivate the development of DL solutions for the domain of fault detection in industrial machines. DL networks are known to cope well with the aspect of high dimensionality in the data [12, 14]. Industrial machinery data typically consists of a plethora of variables including vibration and audio signals, speed measurements, temperature and pressure values, collected over multiple sensors attached to the machines. Additionally, these measurements are often collected at high sampling rates, resulting in huge data volumes. Another important aspect of DL models is that they take advantage of large data volume availability to improve their learning performance [12, 14, 15]. 

**Figure 1.5: learning curve of traditional ML and DL methods with increasing data volume [15].** 

In general, DL models have sophisticated structures that allow them to learn complex patterns in the data. They are designed to automatically learn hierarchical representations in the data, as they involve a great number of processing nodes and learnable parameters, which allows them to capture complex and abstract patterns [12, 14, 15]. 

When it comes to processing time series, DL recurrent networks can model sequential data in a natural way. Specifically, entire sequences of data can be provided as inputs to DL models to 

6 

learn the temporal dependencies between the observations. During the sequential processing, information from previous parts of a sequence becomes available in the processing of future parts, which allows learning patterns from longer periods of time [16, 17]. 

**Figure 1.6: Feature extraction in ML and DL techniques.** 

An essential advantage of DL models is that they can avoid complexity and requirement for domain-specific knowledge arising from manual feature extraction techniques, which are often included in the pipeline of traditional ML approaches [10, 14, 15, 16]. The CNN is a typical example of automatic feature extraction, where initial layers in the network are responsible for detecting low-level features, while the deeper layers learn more complex features by combining information from the lower layers [18]. This hierarchical approach of feature extraction allows 

7 

the model to effectively capture patterns and relationships in the data that might not be apparent to humans.  Another example of automatic feature extraction is the Attention mechanism which evaluates the importance of features in different timesteps, and it distributes larger weights to the most relevant parts of a sequence in terms of predicting the target output [16, 19]. 

Automatic feature extraction in CNNs and Attention can also aid in mitigating noise in the input data [20]. Machinery data can often include noise due to factors like measurement errors, sensor drifting or poor sensor placement, and mistakes during data collection or recording. By being able to automatically extract high-level features from raw input data, DL models can focus more on relevant information, while filtering out irrelevant or noisy parts of the data. 

Regularization is also another concept integrated into DL networks that can have positive effects on noisy data. Regularization techniques discourage a model to focus on specific features, by penalizing the relevant weights during the computation of the loss function [21]. This means that even if the model encounters noise during training, it is less likely to overly focus on such information, which can reduce the risks of overfitting the noisy data. 

Nevertheless, DL models can face their own challenges when it comes to fault prediction and time series. For example, the problem of vanishing gradients in recurrent networks due to the series of multiplications taking place in every timestep can impose some difficulties in retaining information from very long periods of time [2, 22, 23]. 

## **1.3 Motivations** 

There is a plethora of reasons to tackle the problem of fault detection in industrial machinery. Machinery faults are associated with a wide range of potential risks of diverse nature. More specifically, they increase unplanned downtime risks, which can degrade or interrupt a factory’s core processes. Unplanned downtimes caused by machinery failure have severe financial impacts for manufacturers. 

**Figure 1.7: Unplanned downtime financial costs per sector [24].** 

8 

According to [24], the problem is one of the largest causes of productivity loss in modern factories, costing manufacturers billions of dollars every year (figure 1.7). Decreased productivity due to unplanned downtimes can have profound impacts such as customer churn, credibility damage, and burden with penalties associated with delays in delivering products. 

Furthermore, malfunctioning industrial machines can impose serious safety hazards. When a machine operates in faulty states can quickly become dangerous. According to a study [25] conducted for the Finnish manufacturing industry, machine malfunction is one of the most contributing factors to fatal workplace accidents. 

**Figure 1.8: Most contributing factors to fatal workplace accidents in Finnish manufacturing industry [25].** 

Another incentive for tackling the problem is inefficient maintenance, which is one of the main factors influencing unplanned downtimes and safety risks. Increased asset complexity in modern industrial equipment makes maintenance planning a challenge. In many cases, maintenance decisions are still made based on human understanding. In some cases, this leads to suboptimal decisions and ineffective maintenance plans [26]. More precisely, when it comes to maintenance there are three possible outcomes; the machine is being maintained too frequently while is not practically required, the machine is not being maintained frequently enough, increasing the risks of total damage, and finally, a machine is maintained upon demand which is the most efficient alternative. 

An article [27] based on data collected from 421 industrial machines with over two years of operation, reveals that post-failure maintenance involves significantly higher costs for manufacturers. Early fault detection can contribute to efficient maintenance plans and suggest cost-effective maintenance strategies, where industrial machines are maintained only when it is 

9 

required. Furthermore, timely alerts indicating potential failure could assist plant employees in taking preventive measures to increase the life expectancy of the relevant assets. 

**Figure 1.9: maintenance costs per machine unit [27].** 

Finally, in some cases machinery failure can result in catastrophic environmental damage, with dangerous repercussions for societies and ecosystems. An example of this, is the oil and gas industry, where pumps are used for extracting and transporting petroleum and gas products. However, pumps are prone to failure, increasing the risks of environmental accidents such as oil spilling, and therefore they must be closely monitored [28]. 

## **1.4 Problem Definition** 

The complexity of modern industrial machinery driven by technological advancements and the demand for delivering complex features to products, along with intensive workload in harsh environments led by global competition and other economic factors, impose some challenges in the industrial sector. These challenges are related to the increasing risks of machinery failure, resulting in undesirable repercussions. 

Malfunctioning machines impose serious safety hazards, often leading to injuries and fatalities, while they can cause catastrophic environmental accidents. Moreover, unplanned downtimes leading to productivity loss, along with the high maintenance cost of damaged equipment, are among the drivers for manufacturers to search for competent solutions to minimize machinery failure. 

Being able to detect faults in early stages is privileged, since the reliability and availability of industrial machines can be increased. Fault detection in industrial machinery aims to learn fault patterns from a machine's operating process and detect erroneous states that can lead to malfunctions or failures, so that timely alarms can be raised, and counteractions can be planned to prevent further damage. 

The digitalization of manufacturing and the technologies involved in I4.0, provide the means to detect machinery faults and increase process stability. Big Data Analytics, as one of the building blocks of I4.0, utilizes a wide range of ML techniques to facilitate the development of models capable of learning a machine’s behavior through historical data. Therefore, this study attempts to approach the problem with DL models and evaluate their suitability for the domain. 

10 

## **1.5 Objectives** 

This study aims to evaluate the effectiveness of various deep learning (DL) models in addressing fault detection in industrial machinery. More specifically, a series of DL solutions based on Recurrent Neural Networks (RNN), including Gated Cell models, like Long ShortTerm Memory (LSTM) and Gated Recurrent Unit (GRU), are applied to the problem. Additionally, the suitability of architectures that have been typically utilized in Natural Language Processing (NLP) tasks, such as Attention mechanism and Transformer, is explored for the domain, along with Convolutional Neural Networks (CNN) that are typically used for image analysis. 

The models are trained in a supervised manner, based on historical data collected from two widely used components in industrial settings: a rotating machine and an industrial gearbox. The models aim to identify fault patterns in rotating machines using the MaFaulDa dataset and in gearboxes using the China Southeast University (CSU) dataset, to detect faults in new, unseen data. To estimate their reliability, their performance is evaluated based on different metrics such as Confusion Matrix, Precision, Recall and F-score. Finally, based on the implementation and the conducted experiments, this study attempts to answer the following research questions: 

- **RQ1:** How can DL solutions be practically utilized to effectively address the problem? 

- **RQ2:** In terms of effectiveness, how do the explored DL models compare to traditional ML models from previous studies when applied to the same datasets? 

- **RQ3:** In terms of effectiveness, how do the explored DL models compare to each other when applied to the problem? 

## **1.6 Ethics** 

The ethical benefits of fault detection in industrial machines include improved working conditions and reduced safety risks from failures and malfunctions. On the other hand, in many cases, decisions related to maintenance and parts replacement are still made based on the experience and expertise of engineers [1, 26]. 

The integration of data-driven models in decision-making processes may create insecurity for certain professions, such as maintenance and safety engineers. However, it is entirely up to the organization to use AI solutions in a manner where the human aspect is not removed from the decision-making process. Precisely, the results of the AI models can be consulted by human experts to enhance their knowledge in the domain and improve the quality of their decisions. 

## **1.7 Sustainability** 

The application of fault detection methods in industrial machinery has a direct impact on sustainability, aligning closely with several United Nations Sustainable Development Goals (https://www.globalgoals.org/goals/). Fault detection can prevent catastrophic environmental accidents and minimize the risks of potentially threatening situations for societies and ecosystems, contributing to Goal 13 (Climate Action). A characteristic example is the oil and 

11 

gas sector, where defects in oil extraction machinery can lead to disruptive outcomes such as oil spills or emissions of dangerous substances, which are significant concerns under Goal 14 (Life Below Water) and Goal 15 (Life on Land). Furthermore, early fault detection can prevent total damage to a machine's components. Timely alerts allow for scheduled repairs, avoiding unnecessary part replacements that might be needed after a failure. This practice not only increases the lifetime of assets but also leads to more efficient waste management, supporting Goal 12 (Responsible Consumption and Production) by optimizing resource use and reducing waste. Additionally, by enhancing industrial reliability and efficiency, this approach fosters innovation and infrastructure development, directly contributing to Goal 9 (Industry, Innovation, and Infrastructure **).** 

## **1.8 Use of AI Tools** 

This thesis was entirely developed without the use of generative AI tools. All text, coding, implementation, troubleshooting, and data preprocessing were conducted exclusively by the author. The models and preprocessing steps were derived from the author's research, professional experience, expertise gained during the master’s program, and a thorough study of referenced literature. The sole use of AI in this work was Grammarly, applied only for minor editorial revisions and to ensure that the document was free of plagiarism. No AI tools contributed to the content generation, conceptual development, or the creation of images or diagrams in this thesis. 

## **1.9 Thesis Structure** 

This thesis is structured in the following manner: Chapter 2 provides an overview of prior scientific research in the field, with a specific focus on both traditional ML and DL approaches. Chapter 3 outlines the theoretical framework for fault detection methods in the domain, process monitoring techniques, time series analysis, and various concepts related to DL models such as their structures and operating methods. Chapter 4 presents the technical approaches employed for the experimental part of this study, including a description of the datasets used, the preprocessing techniques, and the architectures of the explored DL networks. Chapter 5 reports the results of the assessment methods used to evaluate the performance of the implemented models. In chapter 6, the outcomes of the scientific work are discussed, along with their reflection on the research questions. Finally, chapter 7 summarizes the conclusions drawn from this study, and provides suggestions for future research in the domain. 

12 

## **2. Related Work** 

Several studies have been published over the last years focusing on fault detection in industrial machinery. Solutions based on traditional ML techniques have been more mature in the field, while in more recent research the problem is approached by exploring DL solutions. This section provides an overview of related literature in the domain. For the purposes of this study and more specifically to address the research questions, the reviewed literature is categorized into two sections. To create a basis for comparison with the explored DL models, the first section provides an overview of prior research addressing the problem based on traditional ML techniques, applied on the same datasets. On the other hand, the second section reviews studies that have investigated similar DL architectures for detecting faults in a wider range of industrial equipment. 

## **2.1 Machine Learning Approaches** 

The suitability of the explored DL models for the domain is evaluated in this study based on two datasets that have been used in previous research works. Both datasets consist of multivariate time series representing signals from sensors and other digital instruments. More precisely, the MaFaulDa dataset comprises data from six different states of a rotating machine, with five states representing different types of fault modes and one state the machine’s normal condition. Similarly, the CSU dataset contains data from five different states of an industrial gearbox, with four states representing different types of gear faults, and one state the normal operation of the gearbox without faults in the gears. Chapter 4.3 provides a more comprehensive overview of the datasets. 

## **2.1.1 MaFaulDa Dataset** 

Given that rotating machines are part of around 40% of the total equipment used in modern industrial environments including manufacturing, oil and gas, and power generation [29], they have attracted the attention of several studies in the field. Some of these studies presented in this section, utilized the MaFaulDa dataset and approached the problem of fault detection based on traditional ML techniques, where the primary objective of the models was to identify and classify the diverse fault types of the rotating machine. 

Many of these studies involve time series decomposition and feature engineering techniques in the data preprocessing pipeline, with the goal to reduce irrelevant information from the data and identify the most important features that are likely to predict the states of the rotating machine. Moreover, since the different states in the dataset are not in perfect balance, in many cases oversampling techniques are added to the pipeline to improve classification performance. 

In one of these research works, Sokolovsky et all [30] presented a cost-effective data preprocessing pipeline based on signal analysis, with the objective to optimize classification accuracy for machinery fault modes. The authors used the Empirical Mode Decomposition (EMD) method to reduce noise by decomposing signals into constituent intrinsic components, where the highest frequency components were subtracted from the original signals. To downsize the dimensionality of the input time series, a time-windowing technique was utilized to collect the 50th data point from each sequence, corresponding to a sample rate of 1kHz, over 

13 

the 5 seconds duration of each sequence. Additionally, feature extraction was performed with TSFEL, where recursive features were eliminated with K-fold cross-validation. After being processed, the data was used by two ML models to classify the states of the rotating machine, where the Logistic Regression model achieved an accuracy score of 93% and the CatBoost classifier an accuracy of 90%. 

Feature extraction was also performed by Martins et al. [31] who applied two different techniques, namely Similarity Based Model (SBM) and Kernel Discriminant Analysis (KDA), to investigate their impact on industrial fault detection. Initially, the authors applied Kurtosis and Entropy to derive new statistical features out of the original ones, and utilized the SBM and KDA in two different experiments to evaluate which method provides the most relevant features for discriminating the fault states of the rotating machine. The results showed that SBM improved the classification accuracy of the Random Forest when applied to the original feature space with 120-fold cross validation by 0.41%, achieving an accuracy score of 94.14%, while KDA led to inferior results, decreasing the Random Forest’s accuracy by 0.42%. 

The Multi-Layer Perceptron (MLP) is the most common model in studies using the MaFaulDa dataset. Although MLP is a type of feed-forward neural network, it typically features a simple three-layer architecture consisting of an input, a hidden, and an output layer. Furthermore, MLPs are not designed to process data sequentially and they do not employ advanced concepts like the “memory” of recurrent networks or the automatic feature extraction in CNNs. Therefore, in line with [10], this study follows the taxonomy outlined in figure 1.3, categorizing the MLP as a shallow learning technique and as part of the wider range of traditional ML techniques. 

The MLP classifier was proposed by Lima et al. [32] for detecting fault states in rotating machines. The authors reduced the dimensionality of the signals in the dataset by applying spectral analysis with FFT, and they derived new statistical features out of the original ones, mostly by measuring the central tendencies of the input sequences. After achieving class balance by introducing zero-mean Gaussian noise to generate new sequences, the processed data was used to train the MLP for discriminating a subset of the rotating machine’s states, such as the normal, unbalanced, and misalignment states. In the inference phase, the classifier reached an accuracy score of 93.5%. 

Another study utilizing the MLP was conducted by Pestana-Viana et al. [33], where the authors investigated the influence of feature vector on its performance. To achieve class balance before the training process, the number of normal, unbalanced, and misalignment states was artificially increased by adding Gaussian white noise to the original signals at the level of 10 dB-SNR. The classifier reached a score of 95.8% accuracy in detecting the fault states of the rotating machine by using data from only three accelerometer signals. 

Ali et al. [34] investigated the influence of handling class imbalance in the context of mechanical fault detection. Specifically, oversampling with SMOTE was applied to achieve class balance for the states of the rotating machine, and an MLP was employed to evaluate the effects of class imbalance in terms of classification performance. The experiments showed that when class balance was achieved with SMOTE, the performance of the MLP classifier improved, reaching an accuracy score of 96.2%. 

Finally, in a research work with many technical similarities, Alzghoul et al. [35] utilized an MLP classifier to approach the problem of capturing fault modes in rotating machines, using a 10-fold cross validation technique. Moreover, the authors explored the impacts of oversampling and feature extraction on the performance of the classifier. The experiments revealed that when 

14 

class balance was achieved with SMOTE and features were extracted with the Relief algorithm, the accuracy of the MLP slightly improved, reaching a score of 96.93% with the test data. 

## **2.1.2 CSU Dataset** 

A gearbox is another common type of equipment that can be found in many industrial settings [36]. Industrial gearboxes are frequently integrated into various industrial machines, including wind turbines, milling machines, industrial mixers, and conveyor systems, where they are mainly employed to control their rotational speed. On the other hand , literature on industrial gearboxes in the context of fault detection is relatively sparse compared to that on rotating machines. However, few research studies presented in this section approached the problem using the CSU dataset. 

In one of these studies, Zhao et al. [36] compared the performance of different ML models and a DL GRU network in the context of detecting faults in industrial gearboxes. The authors performed feature extraction with the Wavelet method, while they downsized the representation of the input sequences based on a time-windowing technique, where 512 data points are selected in each sampling window. The processed data was provided to three ML models with the objective to classify the gear faults in the CSU dataset, where the SVM achieved an accuracy score of 94.40%, the KNN 93.20%, and the MLP 90.06%. However, the GRU model outperformed the ML models in the classification task by reaching an accuracy score of 95.80%. 

Pang et al. [37] proposed a fault detection method for industrial gearboxes based on Random Forest classifier. The authors utilized Singular Value Difference Spectrum (SVDS) to denoise and reconstruct the signals in the dataset, by extracting the most important features. The Random Forest was trained using a bootstrap resampling method, where the data overlap degree was increased to enhance its learning ability. The model classified the states of the industrial gearbox with an accuracy score of 90.90% with 0% data overlap degree and 96.30% with 90% data overlap degree. 

Finally, one more study using the CSU dataset to compare the performance of an LSTM network with the one of an SVM classifier is discussed in the next section, in the context of DL architectures. 

## **2.2 Deep Learning Approaches** 

CNNs along with RNNs and their variations, such as LSTM and GRU, are the most frequently employed models in research approaching the problem with DL solutions. However, more recent research suggests a growing interest in using hybrid models, particularly for their ability to combine the strengths of different architectures. Furthermore, the Attention mechanism and Transformers, initially introduced for Natural Language Processing (NLP), have also attracted the authors’ interest in most recent years. This chapter provides an overview of literature in the domain. Initially, studies focusing on single-model DL solutions are discussed, followed by a presentation of hybrid solutions and approaches based on Attention and Transformer. 

Recurrent networks are designed to work with sequential data, such as time series. They are particularly suitable for modeling time dependencies and capturing long-term temporal patterns in the data [2, 31]. The advantage of recurrent networks is that they utilize the concept of “memory”, which allows to process each input in the sequence while maintaining an internal 

15 

state that captures information from parts of the sequence that have been already processed. This internal state is updated with every new input, so the network can adapt its learning based on the context of a sequence [23]. This feature makes them especially appealing for the domain, where data is mostly derived from signals that can be naturally represented as time series. In recent years, there have been several studies exploring the effectiveness of recurrent networks for fault detection in various industrial applications. 

A DL architecture based on GRU was proposed by Nath et al. [29] to detect fault modes in rotating machines in a timely manner, utilizing the MaFaulDa dataset. Initially, the authors reduced the representation size of the input sequences by applying FFT to identify distinctive frequency components, and the downsized data was provided to the model for training. The structure of the model involved an input layer that introduces the input data to the network in a sequential manner, a GRU layer consisting of 100 units, and an output layer with softmax for classifying the states of the rotating machine. The proposed solution achieved an accuracy score of 98.3% in the classification task. 

The LSTM was proposed by Chen at al. [38] to approach the problem of fault detection in industrial gearboxes, utilizing the CSU dataset. The network involved an input layer with size equal to the entire size of an input sequence, five consecutive LSTM layers consisting of 220 neurons each, and a softmax output layer with five neurons, one for each categorical output. The authors used the Particle Swarm Optimization (PSO) algorithm to optimize the model’s learnable parameters and minimize the risk of settling to local minima points. The LSTM was trained at the learning rate of 0.001, and it achieved an accuracy score of 97.44% in discriminating the states of the gearbox. The performance of the proposed model was also compared to an SVM classifier which reached an accuracy score of 88.16%. 

Building on LSTM networks' effectiveness, Cao et al.'s paper [65] proposes an intelligent fault diagnosis method for planetary gearboxes, optimizing an LSTM network with a Modified Particle Swarm Optimization (MPSO) algorithm. Essential features are extracted and selected via a two-sample z test to form a refined feature dataset. The LSTM classifies fault types and the MPSO automates optimization of the learnable parameters. This method achieved an overall accuracy of 99%, outperforming traditional techniques like Entropy Feature Fusion and SVM. 

A comparative study between recurrent networks was conducted by Naren et al. [2] using RNN, LSTM, and GRU. The authors found that all models could learn to detect wind turbine faults. In a related study, Demidova [22] compared the same recurrent models in the context of classifying degradation states of turbofan engines based on severity, with the goal to optimize maintenance schedules. In both studies the GRU and LSTM networks demonstrated better capabilities in classifying the fault states of the wind turbine and the turbofan engine compared to the RNN. Moreover, both studies revealed that LSTM and GRU could deal more effectively with the vanishing gradient problem. In both studies the GRU slightly outperformed the LSTM, which required the lengthiest training, while the RNN could be trained in shorter time. 

A more comprehensive comparison between GRU and LSTM was performed by Mateus et al. [23], in a study where both models were utilized to detect fault conditions in a pulp paper press machine. The models were configured with numerous hyperparameter combinations, and the results showed that both networks could predict the fault modes with high accuracy. The GRU though could work better with both smaller and larger time series samples, and it could learn more efficiently under a wider range of configurations. Additionally, the GRU appeared more immune to the vanishing gradients problem for the given task, reaching an accuracy score of 98%, while it was also faster to train. 

16 

Another type of DL model that has captured the interest of some studies in the domain, is the CNN. Over the years, CNNs have become a widely used tool for image processing tasks due to their strong feature extraction capabilities in 2D vectors [10, 39, 40]. However, research has shown that CNNs can also be effectively applied to 1D time series data, such as those generated by industrial machines. Specifically, CNNs can extract the most relevant features from different parts of a sequence, regardless of their position within the sequence [40]. This is achieved by applying 1D convolutional filters over different parts of the time series, to identify patterns that are relevant to the current task. By extracting the most important information from the data, CNNs can decrease representation size and make time series processing more efficient, while reducing training time [40, 41, 42]. Consequently, there is growing interest in CNNs for the potential benefits they can bring in the domain. 

One notable study examining the suitability of CNNs for the domain was conducted by Guo et al. [39]. This research proposed a hierarchical CNN architecture with adaptive learning to classify bearing faults in a rotating machine. The network was composed of stacked convolutional and pooling layers that extracted the maximum value of the input signals over a spatial window of fixed size, and they were used for fault identification, pattern recognition, and automatic feature extraction. The outputs of the pooling layers were then provided to a fully connected layer with softmax activation to estimate a probability for each sequence to belong to a certain class. The proposed network was able to classify the bearing faults with 98.1% accuracy. 

In another study, Souza et al. [41] employed a CNN to optimize maintenance plans based on forecasting faults in rotating machines, using the MaFaulDa dataset. The proposed model relied on a combination of FFT and a 1D CNN to downsize representation and extract features from three vibration signals. More precisely, the network operated in four phases. Initially, a convolution layer performed parallel processing to generate a set of linear activations. In the second phase, the linear activations (ReLU) were performed to normalize the model’s parameters, while in the third phase, a max-pooling layer extracted the maximum value of each patch in the feature map and modified the output for the next layer. At last, a fully connected layer with softmax was used to classify the machine states with 97.25% accuracy. 

The CNN model was also used by Kiangala et al. [42] for detecting fault states in industrial conveyors. In this study the time series signals were initially provided to a convolutional layer to filter and map each part of the input to a smaller one, while a max-pooling layer generated spatial features and further reduced the size of the feature maps. Moreover, the network involved a flatten layer to convert the pooled feature map to a single dimensional vector, and a fully connected layer to classify the outputs. The proposed solution achieved 100% accuracy in classifying three operating states of the conveyor, including two fault states and the normal. 

Few researchers have compared CNNs and recurrent networks in the context of fault detection in industrial machines. In two of these studies, the authors compared the effectiveness of recurrent networks in capturing the long-term dependencies of the processed signals, with that of CNNs in detecting repeated patterns by extracting local features from consecutive parts of the input signals. Specifically, Lei et al. [43] compared the effectiveness of an LSTM and a CNN network in detecting wind turbine faults, with the LSTM outperforming its counterpart by reaching an accuracy score of 99.8% in the relevant classification task. 

The comparison of the two networks was the main topic of the research conducted by Jansen et al. [3], with the goal to optimize maintenance schedules for industrial machines through fault detection. The authors utilized a CNN that performs local feature extraction with convolutional layers and feature map down-sampling with pooling layers. On the other hand, the LSTM 

17 

contained several layers to address the challenge of long-term dependencies in the input signals and overcome the vanishing gradients problem. In the classification task of detecting 20 fault modes of an industrial cutting machine, the LSTM outperformed the CNN, by reaching an accuracy of 82%, with data that was not initially collected for fault detection. 

While the aforementioned studies focused on comparing CNN and LSTM, Nasser et al. [40] combined the advantages of both networks into a hybrid architecture to address the problem of predictive maintenance by identifying machine faults. In the proposed CNN-LSTM model, the CNN applied fixed-size filters to the input signals and extracted local features, while reducing the size of the feature vector. On the other hand, the LSTM focused on capturing long-term dependencies in the input signals. The CNN network consisted of a convolutional layer with ReLU activation, followed by a max-pooling layer and a dropout layer to prevent overfitting. The outputs of the CNN were fed to the LSTM with 100 units, and finally to a fully connected layer with softmax to provide the classifications. The hybrid solution was compared to a basic RNN and LSTM, and it slightly outperformed them (average F-Score = 0.975) in detecting faults from 10 different machines. 

In a parallel effort, Zhou and Tang [66] proposed a CNN-LSTM architecture for fault diagnosis in rotating machinery. Their model used CNNs to extract local spatial features and LSTMs to learn temporal dependencies from input vibration signals. This hybrid approach achieved a fault diagnosis accuracy of 97.85%, outperforming similar standalone CNN and LSTM models. The model's superior performance was validated through comprehensive evaluations. 

With CNNs and recurrent networks being the most prevalent DL models in related scientific literature, more recent research in the domain investigates the applicability of two models that have emerged as powerful tools in the field of NLP [18]; the Attention mechanism and the Transformer model. The Attention mechanism was introduced to address the problem of information bottleneck in encoder-decoder networks, where the encoder has to compress the entire input sequence into a fixed-size vector. By applying the mechanism, the decoder can choose the relevant parts of the input sequence to focus during decoding. Attention can enhance a model’s robustness by putting more weight on the informative parts of the sequences [44]. Thus, the potential advantages of the mechanism for the domain have been the focus of more recent literature. 

In research conducted by Y. Li [18], an LSTM with Attention was introduced to detect faults in the process of an industrial chemical reactor. The LSTM network adapted to the spatial data structure generated in production lines, while the Attention generated a weight vector that distinguished the crucial information from the redundant in the input signals. The weight vector was combined with the outputs of every LSTM unit to determine the correlation of each input with a fault, and thus inputs that were more relevant for detecting the faults of the reactor were given bigger weights. The proposed solution achieved accuracy scores between 98% and 100% with multiple types of faults, outperforming models from previous studies like SVM and basic LSTM, utilizing the same dataset. 

In a related study, X. Li et al. [45] introduced a DL model based on CNN, LSTM, and Attention mechanism for fault diagnosis in the rolling-element bearings of a rotating machine. Initially, the authors reduced the size of the input signals by applying FFT. Then, a CNN consisting of convolutional and max-pooling layers extracted the spatial features and provided its outputs to the LSTM to capture long-term dependencies in the signals. The Attention accessed past hidden states in the LSTM to locate the most informative parts of the segmented input signals and calculated the weights of the relevant context vector that summarized the importance of each segment in classifying the bearing faults. Finally, a fully connected layer with softmax was 

18 

integrated to provide the classifications. The proposed model was compared to a similar model without the Attention mechanism, which it outperformed with a 97.7% accuracy score. 

The same DL architecture was employed by Xue et al. [46] for detecting wind turbine faults. In the proposed Attention-based CNN-LSTM, the CNN, consisting of convolutional and pooling layers, extracted local features and downsized representation, while the LSTM processed the CNN’s outputs to learn deeper time dependencies in the extracted features. Finally, the Attention focused on capturing important fault-related information in the data and efficiently distributed the weights of the context vector. The proposed network outperformed an XGboost classifier, a basic RNN, and an LSTM in classifying icing and yaw faults in wind turbines with 96.46% and 98.21% accuracy, respectively. Additionally, it appeared to capture the temporal relations in time series more deeply than the other networks. 

The Attention mechanism is also the key element of a Transformer model. Transformers do not rely on recurrent networks for encoding and sequential data processing. Instead, they utilize multiple self-attentions to attend simultaneously all parts of a sequence and compute the attention weights in a parallel way that reflects the relevance of each element’s position in the entire sequence [44]. 

The Transformer model was proposed by Jin et al. [47] for bearing fault diagnosis in rotating machines. The authors initially designed a token sequence generation technique to handle input data in 1D format. The vibration signals were then provided to a Transformer layer consisting of stacked multi-head Attention blocks, to capture the most relevant parts indicating bearing faults. The final block in the Transformer stack involved an MLP to achieve more complex nonlinear mapping. Finally, the feature maps from the Transformer layer were given to a classification layer to one-hot encode the model’s outputs. The Transformer could detect various bearing faults with 98.63% accuracy, outperforming other models like CNN, RNN and LSTM. 

In a cognate study focusing on rotating machines, Nascimento et al. [48] introduced a similar architecture for identifying rotor and bearing faults based on severity level, with the objective of optimizing maintenance plans in industrial manufacturing. The Transformer model was composed of stacked multi-head Attention mechanisms for simultaneously processing multiple subspaces in the input signals, while it employed residual connections to facilitate information flow and prevent the gradients from vanishing. The proposed solution was compared to models from previous studies applied on the same dataset, such as GRU, SVM, CNN, and XGboost, which it outperformed by reaching an accuracy score of 100%. 

The benefits of the Transformer and CNN were combined in a study conducted by Wang et al. [49], with the goal to classify rolling element bearing faults in rotating machines. The authors used a simple transformation method to convert the multivariate time series into 2D images, which were provided to a CNN for feature extraction and representation size reduction. The feature maps from the CNN were utilized by the Transformer, composed of stacked scaled dotproduct Attention modules, to perform deeper analysis by attending more information in parallel. At last, a fully connected layer with softmax was used to classify the bearing faults. The CNN-Transformer model was compared to other CNN-based approaches, which it outperformed with an accuracy score of 99%. 

Finally, the LiteFormer model was introduced by Sun et al. [67] for fault diagnosis in rotating machines. The authors developed a lightweight and efficient Transformer architecture specifically designed to process vibration signal data. The model leveraged convolutional tokenization to enhance feature extraction and employed linear self-attention mechanisms to reduce computational complexity while maintaining high performance. The LiteFormer model 

19 

demonstrated state-of-the-art results, achieving test accuracies of 100%, 99.94%, 99.90%, and 99.82% on various benchmark datasets, including the CWRU, PU, and SEU datasets. This performance significantly surpassed the traditional CNN in both accuracy and computational efficiency. 

Most studies in the field compare proposed DL solutions with only a limited number of traditional ML algorithms, making it challenging to gauge how these solutions stack up against other DL models. A broader evaluation of DL architectures on the same datasets can offer deeper insights into their suitability for fault detection in industrial machinery. However, many studies lack architectural transparency and comprehensive evaluation metrics, obscuring a full understanding of model performance. Additionally, it would be beneficial to include more thorough comparisons with traditional ML methods, not only in terms of accuracy but also concerning data preprocessing, to better assess their relative advantages and limitations. 

20 

## **3. Theoretical Underpinning** 

## **3.1 Fault Detection** 

This section offers a detailed overview of different approaches for fault detection and diagnosis in industrial machines. More specifically, it discusses the theoretical background of these approaches and how they are applied in practice, with focus on data-driven condition monitoring and its role in detecting machinery faults. Finally, the concept of early fault detection is analyzed. 

## **3.1.1 Fault Detection Methods** 

Fault detection methods for industrial machines can be categorized differently, depending on the specific manufacturing domain and the nature of the equipment. In general, such methods are divided in two broad categories: model-based and data-driven methods [50, 51, 52]. The model-based approaches use mathematical models such as parity equations and state observer models, developed based on prior knowledge of a physical system. The core of such a mathematical model consists of theoretical assumptions and empirical estimations about the system's functionality, where actual measurements are compared against theoretical thresholds to identify faults [53]. 

**Figure 3.1: Model-based and data-driven methods for fault detection and diagnosis [52].** 

Data-driven methods represent a modern approach within the field since they enable data mining and ML techniques to build predictive models capable of learning a system’s functional 

21 

behavior by utilizing historical data. While both methods generally achieve reliable diagnostic accuracy, model-based methods necessitate extensive domain-specific knowledge to build accurate models [51, 53]. The fact that they are highly dependent on a system’s knowledge depth, makes them less flexible to be applied to machines other than those sharing the exact same functional specifications. Furthermore, model-based methods are considered labordemanding since it is difficult to accurately determine a model’s physical parameters. 

Conversely, data-driven methods rely on large volumes of data to understand a machine's behavior and do not require in-depth prior knowledge, as they learn directly from data [52]. This aspect makes data-driven methods flexible to apply to a wider range of industrial equipment. Therefore, this study focuses on data-driven methods. 

## **3.1.2 Data-Driven Condition Monitoring** 

A condition monitoring pipeline utilizes technological tools to monitor and evaluate the condition and operational states of industrial machines. Modern condition monitoring methods are designed based on data-driven architectures and they integrate I4.0 technologies. The core of such architecture involves AI models developed to process and analyze data from operating machines and estimate the effects of changes and disruptions in their process [52, 54]. Within a machine’s operating process some changes and events are undesired due to their potential negative impacts. These changes are the faults. 

Data-driven condition monitoring systems ensure industrial machines meet performance specifications by employing measures sensitive to anomalies and faults, and by anticipating tasks to address these issues as they arise. The architecture of a modern data-driven condition monitoring pipeline [55] is illustrated in figure 3.2. 

The architecture consists of two main building blocks: the front end and the back end. The front end is responsible for the real-time condition monitoring of the connected machine. More specifically, the real-time fault detector which can be any type of a single-board computer, collects data at a certain frequency (sampling rate) from the attached machine during its operating process. This real-time data includes readings like temperature, sound, and vibration signals, collected via the fault detector’s sensors. Moreover, the real-time fault detector is accountable for data preprocessing, and it hosts an AI-model (ML or DL) which can classify the states of the monitored machine. 

Furthermore, the real-time detector stores the new data streams generated by the machine in a database, which can be used later by the trainer model in the back end to enhance its learning process and increase accuracy. The real-time fault detector's classification results are relayed to a monitoring system that interprets, visualizes, and nearly instantaneously notifies operators of any changes in the machine's state. This way, immediate actions can be taken, if necessary, to prevent the monitored machine from failing. Additionally, such systems can submit maintenance tickets automatically and they can generate event logs which can be later used by experts to analyze the causes of faults. 

In the back end, an AI model is used as a trainer model. The trainer model is an offline model that enables ML or DL techniques to optimize its fault detection abilities. Periodically, the trainer model collects new data streams stored in the database by the real-time fault detector to initiate a new training process. With increased data availability, the trainer model can potentially improve its classification accuracy, and the most effective model is then employed by the real-time fault detector for immediate fault identification. This way the real-time fault 

22 

detector uses the best trainer AI model independently without the need to communicate with the back-end model in real time, which significantly reduces latencies and network overhead. 

**Figure 3.2: Industrial process monitoring tasks [55].** 

However, this is an example of a data-driven condition monitoring method rather than a norm. Condition monitoring pipelines vary and can be customized based on the nature of the monitored assets and the purpose of their service. In some cases, fault detection is performed in two discrete phases; In the detection phase a binary classification task differentiates normal conditions from anomalies, while in diagnosis phase (isolation and identification) a multi-class classification task categorizes anomalies into fault types [51, 52, 54]. 

## **3.1.3 Early Fault Detection** 

The goal of early fault detection is to identify faults occurring in a machine’s operating process in incipient stages and prevent their progression towards failure. This way, timely indicators can be raised, to perform counteractions against unfavorable situations [9]. Such counteractions include temporary machine shutdown to prevent complete damage, immediate replacement of the compartments that are likely to cause a failure, reconfiguration of settings to isolate faulty components till further action is taken, and operations related to time-effective maintenance. 

In most cases, fault occurrence does not imply immediate failure. Failure in industrial machines is an evolving situation that exhibits early indications, such as rising temperatures, increasing 

23 

vibrations, declining voltage levels, etc., that deteriorate a machine's operating performance [56]. Furthermore, a failure might be the result of one or more deviations from normal behavior. 

In data-driven fault detection, variables such as sensor readings are monitored by the model during a machine’s operation. The idea is that the model learns the fault patterns from the historical data, and it detects a fault’s development at an early stage before it evolves to failure. Specifically, models aim to capture the patterns and trends of these deviations by considering not only the relations between the variables but also the order of observations in time. Therefore, sequential data processing can be particularly effective for early fault identification. 

**Figure 3.3: Early fault detection chronogram [56].** 

## **3.2 Time Series** 

Time series is a collection of consecutive observations in time acquired through periodic measurements, in contrast to cross-sectional data that represent single-point-in-time facts. The temporal aspect of time series suggests potential correlations between observations. Furthermore, time series can be categorized into univariate and multivariate [42]. Univariate time series consist of observations from a single variable collected over a period of time, while multivariate time series are composed of observations from multiple variables. Univariate time series can be expressed as demonstrated below (1): 

**==> picture [100 x 15] intentionally omitted <==**

A variable is a vector representing a set of observations _o_ of size _n_ , over a period _p_ . Thus, multivariate time series can be expressed as follows (2), with _m_ denoting the number of univariate variables in a matrix _M_ collected over _p_ . 

**==> picture [207 x 32] intentionally omitted <==**

24 

Given the complexity of modern industrial machinery, data-driven models primarily utilize multivariate time series, which consist of independent variables representing signals from sensors and other digital instruments. Each variable does not only depend on the other variables, but also on its own past values. 

In supervised fault detection, time series data is fed sequentially into a model to learn fault patterns, enabling it to distinguish these from normal sequences during condition monitoring. Ideally, for the model’s training process, the data is organized in regular time intervals, and it contains labels representing a machine’s state, based on the occurrence of certain events across the sequence’s timeline. 

**Figure 3.4: Fault detection in time series data [18].** 

In the case of multivariate time series, the data consists of multiple univariate time series (𝑋𝑋1, 𝑋𝑋2, 𝑋𝑋3 … , 𝑋𝑋𝑛𝑛) where each univariate series is the product of a sensor or digital instrument attached to a machine’s crucial part [18]. Depending on the machine’s nature and specifications, a sampling period/ frequency is determined ( _L_ ), where a collective portion of observations from each univariate time series (parallel series) is sampled together, comprising a sequence of data points that can be described by the parallel observations. Thus, a sequence can be expressed as the multivariate vector 𝑆𝑆𝐿𝐿 = {𝑉𝑉[𝐿𝐿] 1[, … , 𝑉𝑉] 𝑚𝑚[𝐿𝐿][}][, where the vector of each univariate time series ][𝑉𝑉][𝐿𝐿][=] {𝑜𝑜1, … , 𝑜𝑜𝑘𝑘 _}_ contains observations sampled over _L_ . For the model’s training each 𝑆𝑆𝐿𝐿 is associated with a label _Y_ denoting the sequence’s type (e.g., normal, or faulty), thus the training data can be expressed as _{_ [𝑆𝑆1, 𝑆𝑆2 … , 𝑆𝑆𝑛𝑛] , [𝑌𝑌1, 𝑌𝑌2 … , 𝑌𝑌𝑛𝑛] _}_ , where _n× L_ corresponds to the time series total length. 

25 

At each timestep 𝑇𝑇𝑖𝑖, a sequence 𝑆𝑆𝐿𝐿 is provided as input to the model aiming to learn patterns that indicate faults. This is achieved by capturing the time dependencies between the observations of each univariate vector 𝑉𝑉𝐿𝐿 and at the same time their correlation to observations from other univariate vectors in 𝑆𝑆𝐿𝐿 _._ Finally, for each processed sequence, the model generates an output, indicating whether it is classified as faulty or normal based on the relevant observations. 

## **3.2.1 Feature Extraction** 

Feature extraction is a technique based on extracting interesting features out of the data to improve model performance. The goal is to summarize the most relevant information from the raw input data and provide a more compact representation of reduced dimensionality, while preserving informativeness and meaningful relations in the data. 

In time series, feature extraction involves data analysis to identify specific patterns or trends that are related to the task at hand [57]. These patterns or trends are then transformed into a new set of features, such as statistical or time domain features. 

Feature extraction is especially useful for fault detection by reducing noise impacts in machinery data. It enhances classifier reliability, lowers computational costs, and shortens training times through more compact data representation. Additionally, it can reduce overfitting risks by eliminating redundant data, ensuring decisions are not based on noise. 

**Figure 3.5: Example of statistical feature extraction in time series [57].** 

26 

## **3.3 Artificial Neural Networks** 

Artificial Neural Networks (ANNs) draw inspiration from biological neurons, the fundamental functional units of the human brain. ANNs consist of successive layers, each of them involving a set of artificial neurons interconnected to each other and associated with an activation function. Moreover, they enable a set of learnable parameters to carry information regarding feature importance, and non-linear functions applied elementwise for generating a layer’s output. At a high level, an ANN is a mathematical function that maps an input to an output, and it can be expresses as follows (3): 

**Figure 3.6: Artificial Neural Network structure.** 

𝑦𝑦= 𝑓𝑓𝑎𝑎(𝑥𝑥) (3) 

In the equation, _y_ denotes the network’s predicted output given the input _x_ , and _a_ its learnable parameters, which are constantly finetuned over the training process in respect to a loss function _l_ , quantifying the difference between the actual and expected output. In the process of computing its outputs, the network supports two types of propagation: forward and backward [58]. 

## **Forward Propagation:** 

In forward propagation the network’s inputs {𝑥𝑥1, 𝑥𝑥2 … 𝑥𝑥𝑖𝑖 _}_ are propagated across the hidden layers in forward direction. The neurons in each layer receive the outputs of the interconnected neurons from the previous layer, to transform it and pass it to the next layer. More precisely, in 

27 

each hidden layer _h_ , the output of every neuron can be described by a non-linear function that applies to the weighted sum of its inputs. This can be expressed as follows (4): 

**==> picture [210 x 32] intentionally omitted <==**

In the equation, 𝑦𝑦[ℎ] 𝑘𝑘[ denotes the output of neuron ] _[k ]_[in layer ] _[h]_[, ] _[n ]_[represents the number of neurons ] of the previous hidden layer _h-1_ interconnected to neuron _k_ in _h_ , and 𝑤𝑤[ℎ  ] 𝑗𝑗→𝑘𝑘[ denotes the weights ] applied the information flow from neuron _j_ in _h-1_ to neuron _k_ in the current hidden layer _h_ . Moreover, 𝑏𝑏[ℎ  ] 𝑘𝑘[represents the added bias applied to the sum of inputs of ] _[k]_[, and ] _[f]_[ the neuron’s ] non-linear activation function. Finally, this is a repetitive process until the information flows to the output layer, where the network’s outputs {𝑦𝑦1, … 𝑦𝑦𝑚𝑚 _}_ are generated, with _m_ denoting the number of classes to be predicted in classification tasks. 

## **Loss Function:** 

To estimate the quality of the network’s outputs, at each iteration of the learning process the actual outputs are compared to the target outputs. A loss function is enabled to estimate the deviations of the predicted outputs _y_ from the target outputs 𝑦𝑦� in terms of error. The loss function can vary depending on the nature of the problem (e.g., classification or regression). For classification problems, cross-entropy is a common loss function, which can be formulated as follows (5): 

**==> picture [266 x 14] intentionally omitted <==**

Where 𝑦𝑦𝑖𝑖 denotes the actual output and 𝑦𝑦�𝑖𝑖 the target output for input _i_ . The function multiplies the log of the predicted probability for the target output, and the important aspect of this is that misclassified outputs that were predicted with high confidence (high probability) are heavily penalized [59]. The network’s goal is to minimize the result of the loss function, so that its outputs are as close as possible to the target outputs. 

## **Backward Propagation:** 

ANNs utilize the gradient descent algorithm to find the minimum value of the loss function by discovering the optimal sets of weights and biases. In every iteration, the algorithm adjusts each neuron’s weights by propagating the error backwards to each hidden layer in the network, while simultaneously measuring the impact of each weight on the loss. Moreover, it computes the local gradient of the loss function with regards to the weight vector _w_ and it follows the direction of the descending gradient to find a minimum global error value, which ideally is 0, so that the weight vector can be populated with the optimal values (target vector 𝑤𝑤�). Initially, the algorithm is unaware of the optimal weights, thus it starts by randomizing the vector _w_ , which is gradually optimized at each step it takes towards the minimum value. While attempting to decrease the loss by reaching to the global minimum, the algorithm might encounter local minima. 

28 

**Figure 3.7: Gradient descent convergence process.** 

The local minima are the points in the curve that can “deceive” the algorithm to settle, but they are not as good as the global minimum. They refer to lower error values compared to other preceding or succeeding points in the curve, in previous and later steps respectively. Several training techniques can be applied though to avoid the side effects of local minima for the algorithm. The size of the steps is defined by a parameter called learning rate or learning step _a_ , dictating how fast the algorithm converges to a minimum. Tiny learning steps result in an increasing number of iterations, delaying convergence and prolonging the learning process. On the other hand, big learning steps might lead to chaotic bouncing of the gradients, turning convergence into an impossible task. The concepts of Gradient descent algorithm and backpropagation are described in detail in chapter 6 of Goodfellow’s book [60]. 

## **3.3.1 Activation Functions** 

Activation functions are one of the essential building blocks of an ANN, responsible for introducing non-linearity into the model, allowing it to learn complex patterns. Each neuron in the network is associated with an activation function, which transforms the neuron’s weighted sum of inputs. Specifically, an activation function f is a mathematical operation that processes 𝑛𝑛 the weighted sum of inputs ∑𝑖𝑖=1 𝑥𝑥𝑖𝑖 𝑤𝑤𝑖𝑖 + 𝑏𝑏, where 𝑥𝑥𝑖𝑖 are the inputs, 𝑤𝑤𝑖𝑖 are the corresponding weights, and _b_ is the bias term. The output of this transformation _y_ (6) is then passed on to the next neuron or provided as the final output of the network. 

**==> picture [130 x 17] intentionally omitted <==**

29 

**Figure 3.8: Structure of an artificial neuron.** 

Some of the most common activation functions utilized by ANNs are described below. 

**1. Sigmoid:** transforms a given input _x_ to values within the range of 0 to 1. It is used in models that generate a probability as output and it can be formulated as follows (7): 

**==> picture [103 x 29] intentionally omitted <==**

**2. Tangent (tanh):** or hyperbolic tangent, transforms a given input _x_ to values within the range of -1 to 1. It is used for solving the saturation problem, and it can be formulated as follows (8): 

**==> picture [127 x 29] intentionally omitted <==**

**3. Rectified Linear Unit (ReLU):** is a non-linear activation function that transforms an input _x_ to 0 if its value is negative, or it keeps _x_ intact if its value is positive. The range of its output is from 0 to infinity and it can be formulated as follows (9): 

**==> picture [65 x 27] intentionally omitted <==**

**4. Softmax:** is used for multi-class classification tasks in a network’s output layer to generate the probability distribution for each input instance to belong to a certain class. Softmax can be formulated as follows (10): 

**==> picture [123 x 39] intentionally omitted <==**

Where 𝑧𝑧⃗ represents the input vector to the softmax, _K_ is the number of classes in the task, 𝑒𝑒[z(i)] denotes the exponential function applied to each element _i_ of the input vector, while the denominator represents the normalization term. 

30 

## **3.3.2 Long-Term Dependencies Challenge** 

One way to represent the functionality of an ANN is through a directed graph where each node (neuron) in the graph in is linked to a mathematical operation (activation function). This graph is often referred to as the computational graph of the network. Due to the complex structure of DNNs, this graph can be very deep, especially in recurrent networks where the same operations are applied repetitively at each timestep in long temporal sequences. This raises some computational challenges. An example would be a computational graph with a path imposing repetitive multiplications on a matrix _w_ (weights matrix), which becomes 𝑤𝑤[𝑡𝑡] after _t_ timesteps. Suppose that _m_ has an eigenvalue decomposition expressed as follows (11): 

𝑤𝑤= 𝑣𝑣 𝑑 𝑑𝑑𝑙𝑙(𝜆𝜆)𝑣𝑣[−1] (11) 

which after _t_ timesteps becomes equivalent to 𝑤𝑤[𝑡𝑡] = 𝑣𝑣 𝑑 𝑑𝑑𝑙𝑙(𝜆𝜆)[𝑡𝑡] 𝑣𝑣[−1] , with _V_ representing a transformation matrix that contains the coordinates of the chosen eigenvectors. Eigenvalues 𝜆𝜆 that are greater in magnitude than the absolute value of 1 will explode, or if they are smaller, they will vanish, known as the exploding and vanishing gradients problem respectively [60]. This can be a problem for some DNNs, since the gradients propagated over several layers tend to explode or vanish, with vanishing gradients occurring often with lengthy sequential data. This is due to the exponential decrease of the weights at each timestep during the long-term interactions in the sequence. 

## **3.4 Deep Learning Networks** 

DL includes various sophisticated models with typically more complex structures compared to basic ANNs, since they utilize multiple hidden layers between the input and output layers to learn data representations of multi-level abstraction. The ability of DL networks to learn more complex representations is not only due to their deep structure, but also due to the use of more advanced concepts such as the “memory” in RNNs, filters in CNNs, the Attention mechanism, and more. Finally, this section presents the theoretical foundations of the DL technologies explored in this study for the domain of fault detection in industrial machines. 

## **3.4.1 Recurrent Neural Network** 

RNN is a class of DNN typically associated with sequential data and time series. The RNN consists of an input layer, recurrent hidden layers, and an output layer, where each connection between the neurons corresponds to a set of learnable weights. RNNs operate based on the notion of time since their hidden layers are organized in discrete timesteps. In contrast to a standard NN where the neurons in each hidden layer are interconnected with all neurons in the next hidden layer, an RNN’s neurons are interconnected only with the neurons that belong to the same hidden layer (or hidden state), which represents a timestep. The outputs of all neurons in the hidden layer are then forwarded in a collective manner to the neurons of the next hidden layer (next timestep) through a single-direction connection. 

More specifically, in the forward propagation, at each timestep _t_ , the current hidden state ℎ𝑡𝑡 receives the current input 𝑥𝑥𝑡𝑡 in the form of a vector 𝑈𝑈𝑡𝑡 and it produces an output 𝑜𝑜𝑡𝑡 representing the output of the network at _t_ . The neurons in ℎ𝑡𝑡 transform the input 𝑥𝑥𝑡𝑡 by applying the tangent 

31 

function to the weight matrix _W_ of the previous hidden state ℎ𝑡𝑡−1 and current input vector 𝑈𝑈𝑡𝑡, where a vector _V_ is generated, representing the output of the hidden state (12), which is part of the next hidden state’s input along with the next input 𝑥𝑥𝑡𝑡+1. 

𝑉𝑉ℎ𝑡𝑡 = 𝑡𝑡𝑑𝑑ℎ𝑛𝑛(𝑊𝑊ℎ𝑡𝑡−1 + 𝑈𝑈𝑥𝑥𝑡𝑡) (12) 

**Figure 3.9: RNN structure [2].** 

Thus, the hidden state acts as a memory for calculations taking place in past states, since outputs from previous states are given as input to the current state. This aspect of RNNs makes them especially applicable for sequential data that preserve time dependencies. 

RNNs reduce complexity by using the same parameters for all inputs across hidden states to generate the predicted outputs [2, 62]. However, the model is prone to the vanishing gradients problem. RNNs use backpropagation to compute the gradients of the loss function in respect to the weights, and the gradients tend to become very small while being propagated backwards, resulting in missing information from long-past hidden states. 

## **3.4.2 Long Short-Term Memory Network** 

Long Short-Term Memory network (LSTM) is an advanced type of RNN, also known as Gated Cell network. LSTMs support a more sophisticated way to handle information flow within the network by replacing standard hidden layers with LSTM units. The main building blocks and the functionality of an LSTM unit during forward propagation are discussed below [2, 62]. 

## **Forget Gate:** 

The Forget ( _F_ ) gate decides at each timestep _t_ , which information is still or no longer useful. Information from the previous hidden state ℎ𝑡𝑡−1 and the current input 𝑥𝑥𝑡𝑡 get concatenated and multiplied with weight matrices before passing though the Forget gate. The result of this multiplication is processed by a sigmoid function ( _σ_ ), where values close to 1 are important and thus are kept, while values close to 0 are considered irrelevant and can be forgotten. The update equation of the Forget gate at timestep _t_ is expressed below (13), where 𝑤𝑤𝐹𝐹 represents the weights applied to ℎ𝑡𝑡−1 and 𝑥𝑥𝑡𝑡 as the gate’s inputs, while 𝑏𝑏𝐹𝐹 its added bias. 

**==> picture [167 x 15] intentionally omitted <==**

32 

## **Input Gate:** 

The Input gate ( _I_ ) is responsible for updating the cell with new useful information. Initially, information from ℎ𝑡𝑡−1 and 𝑥𝑥𝑡𝑡 are concatenated and pass through a sigmoid (14), where information close to 1 are more important. 

**==> picture [161 x 15] intentionally omitted <==**

## **Cell state:** 

After the activation of the Input gate, the computation of cell state 𝐶𝐶𝑡𝑡 can be initiated. Initially, the concatenation of ℎ𝑡𝑡−1 and 𝑥𝑥𝑡𝑡 passes also through a tangent function to regulate the network. The tangent activation creates a vector consisting of all possible values from ℎ𝑡𝑡−1 and 𝑥𝑥𝑡𝑡. This vector 𝑣𝑣𝑡𝑡 (15) represents the candidate values that can be applied to the cell state, and it is often referred to as the candidate state. The tangent can effectively distribute the gradients and prevent them from vanishing or exploding. 

**==> picture [184 x 15] intentionally omitted <==**

Next, the vector from the tangent and the output from the sigmoid 𝐼𝐼𝑡𝑡 yielding the important values in 𝑣𝑣𝑡𝑡, are pointwise multiplied. Similarly, the output of the Forget gate 𝐹𝐹𝑡𝑡 is pointwise multiplied with the previous cell state 𝐶𝐶𝑡𝑡−1 , and their products are pointwise added to update the current cell state (16) of the network: 

**==> picture [154 x 15] intentionally omitted <==**

**Figure 3.10 LSTM structure [2].** 

## **Output Gate:** 

Finally, the Output gate ( _O_ ) is responsible for deciding which information from past inputs flows to the next hidden state (LSTM unit) by computing the new hidden state ℎ𝑡𝑡 _._ Here, the 

33 

previous hidden state ℎ𝑡𝑡−1 and current input 𝑥𝑥𝑡𝑡 are concatenated and pass through a sigmoid, and their result is the network’s output at timestep _t_ (17): 

**==> picture [170 x 15] intentionally omitted <==**

The cell state is passed to a tangent function, and its output is pointwise multiplied with the output of the sigmoid. Their product is a vector that defines the current hidden state ℎ𝑡𝑡 (18), and it contains the relevant information to transfer to the next hidden state ℎ𝑡𝑡+1. Thus, the current cell and hidden states are always transferred to the next hidden state. 

**==> picture [134 x 15] intentionally omitted <==**

## **3.4.3 Gated Recurrent Unit Network** 

The Gated Recurrent Unit network (GRU) is another advanced variation of RNN and similarly to the LSTM it utilizes the concept of gates. Specifically, a GRU unit has a simpler structure compared to the LSTM, since it consists of two gates, and it does not preserve a cell state. The functionality of a GRU unit during forward propagation is described below [2, 62]. 

## **Reset Gate:** 

The first gate is the Reset gate ( _R_ ) which behaves similarly to the Forget and Input gate of the LSTM, since it determines how much of the past information is irrelevant and therefore can be forgotten. Here, a sigmoid function ( _σ_ ) is applied to the concatenation of the previous hidden state ℎ𝑡𝑡−1 that contains information from previous timesteps, with the current input 𝑥𝑥𝑡𝑡. The update equation of the Forget gate at timestep _t_ is expressed below (19), where 𝑊𝑊𝐹𝐹 represents the weights applied to ℎ𝑡𝑡−1 and 𝑥𝑥𝑡𝑡 as the gate’s inputs, and 𝑏𝑏𝑅𝑅 its added bias. 

**==> picture [172 x 14] intentionally omitted <==**

## **Update Gate:** 

The other gate is the Update gate ( _Z_ ) (20), which decides how much of the past information is relevant and can be transferred to future hidden states. 

**==> picture [172 x 15] intentionally omitted <==**

## **Memory:** 

A GRU unit utilizes the memory to acquire relevant information from previous hidden states. At first, the output of the Reset gate 𝑅𝑅𝑡𝑡 is pointwise multiplied with the information from the previous hidden state ℎ𝑡𝑡−1, and their product is concatenated with the current input 𝑥𝑥𝑡𝑡, to pass through a tangent function after applying the relevant weights 𝑤𝑤𝑚𝑚. The outcome of this operation is a vector 𝑣𝑣𝑡𝑡 (21) that, similarly to the LSTM’s cell state _,_ represents the candidate memory vector (or current memory) containing all possible values that can be passed over to the next hidden state ℎ . 𝑡𝑡+1 

**==> picture [222 x 14] intentionally omitted <==**

34 

Alike the LSTM’s Output gate, the memory also determines which information from the current memory 𝑣𝑣𝑡𝑡 and past hidden states are transferred to the next hidden state. To achieve this, the current memory vector 𝑣𝑣𝑡𝑡 is pointwise multiplied with the output of the Update gate 𝑍𝑍𝑡𝑡, and the previous hidden ℎ𝑡𝑡−1 state with 1-𝑍𝑍𝑡𝑡. Their products are then added to form the current hiddenstate vector (final memory vector) ℎ𝑡𝑡 (22) which is finally passed to the next hidden state ℎ𝑡𝑡+1. 

**==> picture [188 x 15] intentionally omitted <==**

**----- Start of picture text -----**<br>
ℎ𝑡𝑡 = (1 −𝑍𝑍𝑡𝑡) ⊙ℎ𝑡𝑡−1 + 𝑍𝑍𝑡𝑡 ⊙𝑣𝑣𝑡𝑡 (22)<br>**----- End of picture text -----**<br>


**Figure 3.11: GRU structure [2].** 

## **3.4.4 Convolutional Neural Network** 

Convolutional Neural Networks (CNN) are a type of DNN operating by applying convolutions to images or sequential data, and they have strong generalization capabilities [42, 43, 50]. Initially, they extract local features from the input data, and they combine them in different layers to extract important features. CNNs operate based on an input layer, a convolutional and a pooling layer, and finally, a fully connected layer for generating the network’s outputs. 

## **Input layer:** 

The input layer can be presented either in a two-dimensional (2D) manner such as timefrequency spectrum, or in a single-dimensional (1D) manner such as time series. The input representation can be denoted as 𝑋𝑋∈𝑅𝑅[𝐴][𝐴𝐴] , where _A_ and _B_ define the dimensions of the input data. 

## **Convolutional layer:** 

The convolution layer _C_ extracts local features from the input _x_ . The input data here is convoluted by a filter (convolution kernel) to generate the layer’s output which is called the feature map. The equation for calculating the convolution layer’s output is demonstrated below 

35 

(23), where _f_ represents the activation function utilized by the neurons in the layer (usually ReLU), ’*’ denotes the convolutional computation, while 𝑤𝑤𝑖𝑖 represents the relevant weights applied to the input data, and 𝑏𝑏𝑖𝑖 is the additive bias of the filter _i_ . 

𝐶𝐶𝑖𝑖 = 𝑓𝑓(𝑥𝑥∗𝑤𝑤𝑖𝑖 + 𝑏𝑏𝑖𝑖) (23) 

**Figure 3.12: Example of single dimensional convolution for time series.** 

## **Pooling layer:** 

The next layer in the network is a pooling layer _P_ , which performs sampling and reduces the dimensionality of the feature map derived in the convolution layer. The pooling layer extracts only essential information in this phase, which mitigates the risk of dimensionality problems, minimizes overfitting, and reduces training time [10, 50]. Two common pooling methods used with the CNNs are the max and average pooling. The max-pooling method extracts the maximum values from every patch in the feature map, and its mathematical representation is demonstrated below (24), where _M_ denotes the polling matrix and _C_ , _D_ its dimensions. 

**==> picture [120 x 25] intentionally omitted <==**

**Figure 3.13: Example of max-pooling operation for time series.** 

On the other hand, the average pooling method (25) extracts the average values from every patch in the feature map. 

**==> picture [139 x 25] intentionally omitted <==**

36 

**Figure 3.14: Example of average pooling operation for time series.** 

## **Fully connected layer:** 

The fully connected layer consists of interconnected neurons which map the distributed feature representation to the sample target space. The inputs of the network are the filter matrix from the pooling layer, which is eventually converted to a different representation, such as a column or a row. Finally, an output layer generates the predicted output values of the entire network. The mathematical representation of the fully connected layer is demonstrated below (26), where 𝑃𝑃𝑖𝑖 denotes the input of the network and 𝑂𝑂(𝑃𝑃𝑖𝑖) its output. Moreover, _w_ denotes the corresponding weights applied to the input, _b_ the additive bias, while _f_ represents the activation function of the neurons in the layer, for example a softmax to normalize the outputs and ensure that the total of all output probabilities is equal to 1 [42]. 

## 𝑂𝑂(𝑃𝑃𝑖𝑖) = 𝑓𝑓(𝑤𝑤· 𝑃𝑃𝑖𝑖 + 𝑏𝑏) (26) 

**Figure 3.15: CNN structure [10].** 

## **3.4.5 Attention Mechanism** 

The Attention model has been mainly used in NLP-related tasks, and it is the evolution of the encoder-decoder model, since it utilizes an Attention layer to enhance performance with longterm dependencies in sequential data. In a standard encoder-decoder network, the encoder is responsible for compressing all contextual information of the input sequences to a single, fixed- 

37 

length vector based on its last hidden state. This often results in missing information during decoding, since it becomes challenging for the encoder to compress all necessary contextual information in the vector, especially when processing lengthy sequential inputs. 

The main difference of the Attention mechanism with the standard encoder-decoder model is that it calculates multiple alignment vectors at each timestep of the decoder. This allows the Attention mechanism to access all past hidden states from the encoder, rather than the last. Instead of treating the entire input sequence with the same importance, the Attention calculates alignment scores for parts of the sequence, based on their contribution to predicting the target. These alignment vectors are then concatenated at each timestep to form the context vector, and thus the decoder can access the most important parts of the entire sequence [44, 61]. 

**Figure 3.16: Attention mechanism [61].** 

## **Encoder:** 

The encoder, which can be any type of RNN, processes the sequential input 𝑥𝑥= (𝑥𝑥1, … , 𝑥𝑥𝑛𝑛) at each timestep, and it computes the hidden state ℎ𝑡𝑡 based on the current input and previous hidden state (27), which is eventually converted to a vector of continuous representation. 

**==> picture [173 x 14] intentionally omitted <==**

## **Attention:** 

In the Attention layer the context vector 𝑐𝑐𝑡𝑡 is updated for every combination of the encoder’s timestep _k_ , and the decoder’s timestep _t_ , by calculating an alignment score for every part of the input sequence (28). In the following equation, 𝑤𝑤𝐸𝐸 denotes the weights applied to the current hidden state of the encoder, 𝑤𝑤𝐶𝐶 denote the weights applied to the past hidden state of the 

38 

decoder, and 𝑤𝑤𝐴𝐴 represents the weights of the function _f_ , applied for the computation of the alignment scores. 

**==> picture [209 x 15] intentionally omitted <==**

All weights here represent the Attention weights which are eventually normalized and capture the significance of the input at timestep _k,_ for decoding the output at timestep _t_ . The context vector at timestep _t_ is computed as the sum of all hidden states of the encoder _n_ , weighted by the relevant alignment scores (29). 

**==> picture [136 x 32] intentionally omitted <==**

## **Decoder:** 

Similarly to the encoder, the decoder can be of any RNN type. Given the input _x_ , the decoder generates an output 𝑦𝑦 at each timestep. First it calculates its current hidden state 𝑠𝑠𝑡𝑡 (30) by taking into account its previous hidden state 𝑠𝑠𝑡𝑡−1, the previously generated output 𝑦𝑦𝑡𝑡−1, and the context vector 𝑐𝑐𝑡𝑡. The decoder’s responsibility is to put more attention to the parts of the sequence that have been assigned higher alignment scores in 𝑐𝑐𝑡𝑡 to produce the outputs. 

**==> picture [130 x 15] intentionally omitted <==**

Finally, a mathematical function (e.g., softmax) is applied to the hidden state of the decoder at each timestep to calculate the network’s output 𝑦𝑦𝑡𝑡. 

## **3.4.6 Transformer** 

Another DL model that was introduced by [44] for NLP tasks is the Transformer model. Similarly to Attention, the idea of the Transformer is based on the encoder-decoder architecture, where both networks are comprised of multiple Attention mechanisms instead of RNNs. The encoder _E_ transforms the input of sequential data into a continuous representation of elements 𝑥𝑥= (𝑥𝑥1, … , 𝑥𝑥𝑚𝑚). For every element in _x_ , the decoder _D_ generates an output 𝑦𝑦𝑡𝑡 = (𝑦𝑦1, … , 𝑦𝑦𝑚𝑚), considered by the model for predicting the next output sequence 𝑦𝑦𝑡𝑡+1. 

## **Encoder:** 

The encoder consists of a stack of _N_ identical layers, with each layer being composed of two sublayers: a multi-head Attention mechanism as first sub-layer, and a position-wise feed forward network as second. In each sublayer, there is a residual connection _r_ , which allows data to flow into latter parts in the structure by skipping some layers. Furthermore, a normalization layer is applied to the outputs of the two sublayers, to normalize the summed inputs (outputs of the sublayers) and eliminate new dependencies between the data. The output of each sublayer can be expressed with the following equation (31), where _x_ denotes the input, and _subLayer(x)_ represents the relevant function applied in each sublayer. 

𝑂𝑂= 𝑛𝑛𝑜𝑜𝑎𝑎𝑚𝑚𝑛𝑛𝑑𝑑𝑦𝑦𝑒𝑒𝑎𝑎(𝑥𝑥+  𝑠 𝑏𝑏𝑛𝑛𝑑𝑑𝑦𝑦𝑒𝑒𝑎𝑎(𝑥𝑥)) (31) 

39 

**Figure 3.17: Transformer model structure [44].** 

## **Muti-head Attention:** 

The multi-head Attention mechanism is based on the parallel operation of multiple Scaled-DotProduct Attention mechanisms. A Scaled-Dot-Product Attention is based on three vectors, representing queries ( _Q_ ), and sets of keys ( _K_ ) and values ( _V_ ), mapped to the relevant outputs. Each output is a weighted sum of the values _V_ , where each value has an associated weight calculated through a compatibility function of the query and the relevant key. The model’s inputs are composed of _Q_ and _V_ vectors of dimensionality 𝑑𝑑𝐾𝐾 and  𝑑𝑑𝑉𝑉 respectively. The dot products of a query with all keys are computed and divided by a scaling factor ( Jd 𝐾𝐾), and a 

40 

softmax function is applied to derive the relevant weights of the values. The mechanism’s output can be expressed as shown below (32). 

𝑄𝑄𝐾𝐾[𝑇𝑇] attention(Q,K,V) = softmax ~~(zz)~~ 𝐾𝐾 V (32) 

**Figure 3.18: Multi-head Attention mechanism diagram [44].** 

Multi-head Attention enables parallel execution of multiple Scaled-Dot-Product Attention models (heads), based on a number of parallel projections _h_ of _Q_ , _K_ , and V, providing access to information from different representation subspaces at different positions simultaneously. Thus, the equation for the model’s output can be expressed as demonstrated below (33), where each head in the stack is denoted as ℎ𝑒𝑒𝑑 𝑖𝑖[= ][𝑑𝑑𝑡][𝑒𝑒𝑛𝑛𝑡𝑡𝑑𝑑𝑜𝑜𝑛𝑛(𝑄𝑄] 𝑖𝑖[𝑤𝑤] 𝑄𝑄[, 𝐾𝐾] 𝑖𝑖[𝑤𝑤] 𝐾𝐾[, 𝑉𝑉] 𝑖𝑖[𝑤𝑤] 𝑉𝑉[)][ with ] _[w ]_[repressing ] the learnable weights applied to each vector. 

𝑚𝑚𝑠𝑠𝑙𝑙𝑡𝑡𝑑𝑑𝑚𝑚𝑒𝑒𝑑 𝑚𝑚𝑡 𝑒𝑒𝑛𝑛𝑡𝑡𝑑𝑑𝑜𝑜𝑛𝑛(𝑄𝑄, 𝐾𝐾, 𝑉𝑉) = concat(ℎ𝑒𝑒𝑑 1 … ℎ𝑒𝑒𝑑 ℎ[)𝑤𝑤][𝑜][𝑡𝑡𝑝𝑝𝑜𝑜𝑡𝑡][ (33) ] 

## **Decoder:** 

Similarly to the encoder, the decoder consists of a stack of _N_ identical layers where each layer is composed of three sublayers. Compared to the encoder, a decoder layer enables an additional sublayer to apply the multi-head Attention mechanism to the outputs of the encoder. A decoder layer also supports residual connections to allow dataflow overpassing other layers in the path, and it utilizes a simple normalization layer like the encoder. The first sublayer consists of a multi-head Attention, which is modified to perform a masking operation for preventing positions from attending to subsequent positions. Due to the output embeddings being shifted to the right by one position, and in combination with masking, it is ensured that forecasts for element _i_ are only dependent on previous outputs generated for elements up to the _i-1_ element. 

Finally, there are variations of Transformer models where the encoder or decoder network is not utilized, depending on the nature of the task at hand. For example, decoders are particularly suitable for generative tasks, such as language translation. 

41 

## **3.5 Evaluation Metrics** 

To assess the performance of ML models in classification tasks, several metrics can be used. Performance metrics are integral to every ML pipeline, and they are used to evaluate how well an inference model has learned the relevant task. This chapter describes the most common metrics used to assess fault detection classifiers. 

## **3.5.1 Confusion Matrix** 

Confusion matrix is a simple tabular representation used to assess model performance in classification tasks. The size of the table depends on the number of classes in the data and each cell provides the number of the correctly classified and misclassified cases for each class. The rows of the table represent the predicted values, while the columns refer to the ground truth values. The following table demonstrates the confusion matrix for a binary classification task (positive-negative classes) where _TP_ denotes the number of cases correctly classified as positive and _TN_ corresponds to the number of correctly classified negative cases. Finally, _FP_ is the number of cases incorrectly classified as positive while being negative, and _FN_ is the number of cases incorrectly classified as negative while being positive. 

**Figure 3.19: Confusion Matrix for binary classification task.** 

## **3.5.2 Accuracy** 

Accuracy is an assessment metric that provides the percentage of the correctly predicted instances ( _TP+TN_ ), out of the total number of instances in the dataset for which the model makes a prediction (correctly classified + misclassified). Accuracy score can be formulated as follows (34): 

**==> picture [202 x 29] intentionally omitted <==**

42 

## **3.5.3 Precision** 

Precision measures how confident we can be that an instance predicted as positive by the model is actually positive. Specifically, in fault prediction, precision calculates the percentage of correctly identified faulty cases out of all cases that the model predicted as faulty (positives). Precision can be formulated as follows (35): 

**==> picture [138 x 30] intentionally omitted <==**

Where _TP_ is the number of fault instances that were correctly classified as fault cases and _FP_ the number of healthy state instances (negatives) that were incorrectly classified as faulty. 

## **3.5.4 Recall** 

The idea of recall is to measure the portion of positive cases captured by the model. More precisely, recall measures the percentage of faulty cases that were correctly classified among all actual faulty cases in the data sample, including the ones that were misclassified as normal. Recall can be formulated as follows (36): 

**==> picture [125 x 30] intentionally omitted <==**

Where _TP_ is the number of faulty cases that were correctly classified and _TN_ the number of actual faulty cases misclassified healthy. 

## **3.5.5 F-Score** 

The F-score, or F-1 score, is the harmonic mean between precision and recall. This metric provides a more balanced evaluation of a model's performance, especially when dealing with imbalanced datasets, since both false positives and false negatives are taken into account, offering a more accurate picture of a model's performance on all classes. The F-score ranges from 0 to 1, where values close to 1 indicate high predictive quality. The F-score can be formulated as follows (37): 

**==> picture [197 x 30] intentionally omitted <==**

## **3.5.6 ROC-AUC** 

The ROC-AUC is a 2-dimensional graph that shows how well a classifier can distinguish between different classes. The AUC score (38) evaluates the classifier based on two key metrics: sensitivity (or True Positive Rate) (39), which indicates the proportion of correctly classified positive samples, and specificity (or True Negative Rate) (40), which indicates the proportion of correctly classified negative samples. Specificity is used to calculate the False 

43 

Positive Rate (41). In the ROC-AUC graph, the False Positive Rate is plotted on the x-axis, while the True Positive Rate is plotted on the y-axis. 

**==> picture [227 x 13] intentionally omitted <==**

**==> picture [151 x 71] intentionally omitted <==**

**==> picture [145 x 14] intentionally omitted <==**

When AUC = 1, the classifier is perfectly capable of distinguishing among the classes, while if AUC ≤ 0.5, the classifier cannot differentiate between positive and negative classes better than a random classifier. Finally, values in between (0.5 < AUC < 1) can be interpreted as the probability that the classifier can successfully discriminate between positive and negative instances, and vice versa. 

44 

## **4. Implementation** 

## **4.1 Pipeline** 

For the experimental part of this study the CRISP-DM approach is followed. 

**Figure 4.1: CRISP-DM phases.** 

In the business understanding phase, the problem is formulated and analyzed, along with the motivations to perform further research on the topic. This phase also involves the domain understanding of the machines from which the relevant datasets are collected, along with the understanding of their structure, operating process, and serving purpose. The data understanding phase involves a detailed exploration of the two datasets used in the experiments, with the goal to determine suitable preparation techniques in the next phase, before providing the data to the models. During the modeling phase, various DL models are developed, trained, and tested on two classification tasks, while their performance is evaluated based on different metrics during the evaluation phase. Finally, for deployment, the model demonstrating the best performance in the classification tasks is recommended. 

## **4.2 Experimental Environment** 

The experimental part of this work was conducted in _TensorFlow 2.6_ environment, and the models were developed in Python 3.9 programming language, using the _Keras_ framework. Moreover, for data exploration and preparation various ML libraries and packages were utilized, such as _SciPy_ , _scikit-learn_ , _Matplotlib_ and _Seaborn_ . Finally, after the development phase, all models were trained in the cloud, on a node with 12 CPUs 2.60GHz, 32GB RAM, utilizing the GPU runtime of an NVIDIA A100-SXM4. 

## **4.3 Datasets** 

For the experimental part of this work two datasets collected from two different types of industrial equipment, a rotating machine and an industrial gearbox, were used. Three criteria guided the selection of these datasets. Firstly, both datasets are publicly available and free from legal constraints or limitations on their use. Secondly, both datasets cover various fault types in addition to normal states and include detailed descriptions of each fault type, as well as 

45 

information about the data acquisition process. Finally, the datasets were selected due to the prevalence of the two types of industrial equipment in various industrial settings. 

## **4.3.1 MaFaulDa Dataset** 

The MaFaulDa dataset [63] comprises six CSV format files containing 1,951 multivariate time series. These sequences represent six different operating states of a rotating machine, as detailed in table 4.1. This includes five states with various fault types and one state representing normal operation. The data was collected from multiple sensors and instruments installed on SpectraQuest’s Machinery Fault Simulator (MFS), with 8 descriptive features (parallel series) presented in table 4.2. Each sequence was recorded at a 50 kHz sampling rate over a duration of 5 seconds, resulting in a total of 250,000 data points capturing sound, vibration, and speed signals. Measurements were taken using six accelerometers for vibrations, a tachometer for the machine’s rotational speed, and a microphone for recording machine sounds under different conditions. More information on the data acquisition methods and experimental setup can be found in the Online Machinery Fault database [63]. Finally, figure 4.3 illustrates the parts of the machine to which the equipment was attached. 

|**State**|**Description**|**Number of**<br>**Sequences**|**Fault Generation**<br>**Approach**|
|---|---|---|---|
|Normal|Normal state without detected faults. Fixed<br>Rotation speed 737 to 3686 rpm|49|-|
|Imbalance|Imbalance faults occurring when the load<br>of the rotor is unequally distributed in the<br>angular direction of the shaft, resulting in<br>vibrations that prevent rotation frequencies<br>above 3300 rpm|333|Radial load weights<br>from 6 g to 35 g<br>attached to the rotor|
|Horizontal|Horizontal misalignment faults occurring<br>when the shaft is horizontally displaced|197|Horizontal motor<br>shifting of 0.5, 1.0, 1.5<br>and 2.0 mm|
|Vertical|Vertical misalignment faults occurring<br>when the shaft is vertically displaced|301|Vertical motor shifting<br>of 0.51, 0.63, 1.27, 1.40,<br>1.78 and 1.90 mm|
|Underhang|Bearing faults occurring in the inner rolling<br>element (location 3 in figure 4.3)|558|Radial load weights<br>from 0 g to 35 g<br>attached to the rotor|
|Overhang|Bearing faults occurring in the outer rolling<br>element (location 7 in figure 4.3)|513|Radial load weights<br>from 0 g to 35 g<br>attached to the rotor|



**Table 4.1: Rotating machinery states.** 

46 

|**Feature**<br>**Id**|**Acquisition**<br>**Device**|**Device**<br>**Type**|**Device**<br>**Model**|**Number**|**Measurements**|
|---|---|---|---|---|---|
|F1|Microphone|Microphone|Shure<br>SM81|1|Sound signal|
|F2, F3, F4|Underhang<br>bearing<br>accelerometer|Sensor|IMI<br>601A01|3|Vibration signals on radial,<br>axial, and tangential directions,<br>one sensor at each direction|
|F5, F6, F7|Overhang<br>bearing<br>accelerometer|Sensor|IMI<br>604B31|3|Vibration signals on radial,<br>axial, and tangential directions,<br>one sensor at each direction|
|F8|Analog<br>tachometer|Instrument|MT-190|1|Rotation speed signal|



**Table 4.2: MaFaulDa descriptive features/ parallel series.** 

**Figure 4.2: Experimental setup for generating the MaFaulDa dataset [41].** 

**Figure 4.3: Rotating machine: 1. Motor, 2. Coupler, 3. Inner rolling element, 4. Underhand accelerometers, 5. Rotor, 6. Shaft, 7. Outer rolling element, 8. Overhang accelerometers [41].** 

47 

## **4.3.2 CSU Dataset** 

The CSU dataset [64] from Southeast University of China contains vibration signals collected from a gearbox as part of the DDS Simulator’s test rig. The dataset consists of CSV format files, where each file contains multivariate time series representing an operating state of the gearbox. The operating states include the normal state of the gearbox and four states with different gear fault types, as demonstrated in Table 4.3. The data was collected from 8 sensors (8 parallel series), as presented in Table 4.4, under a rotating speed-load configuration of 20 Hz - 0 V. Each CSV file in the dataset contains 1048560 data points. For the experiments in this work, the data can be segmented into cycles or sequences of 1,024 data points each (except for the final, incomplete sequence), corresponding to a 1-second duration at a sampling rate of 1,024 Hz. This results in 1023 complete cycles per state, where each cycle corresponds to a sequence of 1024 data points with 8 parallel series (1024 measurements from 8 sensors to complete a cycle). 

|**State**|**Description**|**Number of**<br>**Sequences**|
|---|---|---|
|Normal|Normal state of the gearbox without faults in the gears|1023|
|Chipped|A crack has occurred in one of the teeth (a)|1023|
|Root|A crack has occurred at the root of one of the teeth (b)|1023|
|Miss|A tooth is missing or has broken off (c)|1023|
|Surface|Wear on the surface of the gear (d)|1023|



**Table 4.3: Gearbox states.** 

**Figure 4.4: Types of gear faults.** 

48 

|**Feature**<br>**Id**|**Acquisition**<br>**Device / Sensor**|**Model**|**Measurements**|
|---|---|---|---|
|F1|Vibrating|608A11|Motor vibrations|
|F2|Vibrating|608A11|Vibrations of the planetary gearbox in x (depth) direction|
|F3|Vibrating|608A11|Vibrations of the planetary gearbox in y (width) direction|
|F4|Vibrating|608A11|Vibrations of the planetary gearbox in z (height) direction|
|F5|Torque|FT293|Motor torque / rotational force|
|F6|Vibrating|608A11|Vibrations of the parallel gearbox in x (depth) direction|
|F7|Vibrating|608A11|Vibrations of the parallel gearbox in y (width) direction|
|F8|Vibrating|608A11|Vibrations of the parallel gearbox in z (height) direction|



**Table 4.4: CSU descriptive features/ parallel series.** 

**Figure 4.5: Experimental setup for generating the CSU dataset [36].** 

49 

## **4.3.3 Exploration and Preprocessing** 

## **MaFaulDa Dataset** 

The following diagram demonstrates the exploration and preprocessing steps for the MafaulDa dataset, before providing the data to the DL models for the training process. 

**Figure 4.6: Data exploration and preprocessing phases of the MaFaulDa dataset.** 

In the MaFaulDa dataset the time series contain only numerical values, and they are organized in different CSV files, where each file contains sequences representing a state of the rotating machine. In the integration step, the CSV files are acquired from the online database and an integration model is built to import them to the experimental environment, while in the labeling step, a class label is assigned to each sequence based on its source file. In the exploration step the class distribution is verified, in terms of number of sequences per state, along with the absence of missing values. 

**Figure 4.7: MaFaulDa dataset class balance.** 

Moreover, in the same step, the scale of the features in all states is explored. To achieve this, the first sequence from each state is selected, and the relevant _Time-Voltage_ graph is plotted for every state. The graph demonstrates the values of all 250000 data points collected over 5 seconds for each state. Each data point corresponds to a point in time 𝑝𝑝𝑡𝑡 where the values _(V)_ 

50 

of all features _{_ 𝐹𝐹1𝑡𝑡 _…_ 𝐹𝐹8𝑡𝑡 _}_ are gathered, and each feature in the graph is represented by a different color. The analysis verifies that the dataset contains a great portion of data points due to the high sampling rate, where 50000 data points were collected per second (50 kHz). Furthermore, it becomes evident from the graphs that the values of the descriptive features are in different scales due to the diverse nature of devices employed to collect the data. 

**Figure 4.8: Feature scale analysis for the MaFaulDa dataset.** 

To avoid lengthy training processes for the models and remain within the available range of computational power, data resampling is performed in the transformation step. To decide the optimal resampling rate without distorting data quality, it is necessary to transfer the time series from the time domain to the frequency domain. To achieve this, the dataset’s frequency spectrum is analyzed by applying the Fast Fourier Transform (FFT) method ( _rfft_ module from _SciPy_ ), where the original signals are decomposed and converted into individual spectral elements. This method uses a folding frequency at 25 kHz (Nyquist frequency sampling), where only the positive frequencies are kept since, due to signal conjugate symmetry, the negative spectrum becomes redundant. By applying the FFT, the number of data points of each sequence is reduced to half (125000). 

51 

**Figure 4.9: Positive spectrum of signals at the original sample rate of 50kHz.** 

52 

Moreover, the FFT analysis reveals that signal frequencies in all sequences are strongly skewed within two frequency bands, 0 to 1 kHz, and 22 to 23 kHz, with the signals at the first band to demonstrate greater strength in all states of the rotating machine. Thus, FFT resampling is performed at a rate of 1 kHz, which reduces the initial number of data points collected at 50 kHz to 5000. Moreover, by only considering the positive spectrum of frequencies, the number of data points in each sequence is further reduced to 2500. By downsizing the number of data points, each sequence can be denoted as a tensor of size <1, 2500, 8>. Similar resampling approaches with FFT were performed in other studies using the same dataset [29, 30, 32, 41], which is worth noting for its relevance and consistency. 

Furthermore, to aid the gradient descent algorithm to converge faster, data rescaling is performed through normalization. Specifically, the Min-Max scaler from _scikit-learn_ is applied to the raw values of each feature as formulated in (42), where _v_ represents a raw value, 𝑣𝑣𝑚𝑚𝑖𝑖𝑛𝑛 the minimum value of the feature, and 𝑣𝑣𝑚𝑚𝑎𝑎𝑥𝑥 the maximum. 

**==> picture [166 x 32] intentionally omitted <==**

## **CSU Dataset** 

The following diagram demonstrates the exploration and preprocessing steps for the CSU dataset, before providing the time series to the DL models for the training process. 

**Figure 4.10: Data exploration and preprocessing phases of the CSU dataset.** 

Each file in the CSU dataset contains 1023 sequences with 1024 data points each, representing a gearbox state under different conditions. Similarly to the previous dataset, the CSV files are initially integrated to the experimental environment as part of the data integration step, and a class label is assigned to each sequence in the data labeling step. During the exploration step, the class balance is verified (figure 4.11), along with the absence of missing values, and the scale of the features in all states is explored. 

To achieve this, each state’s first sequence is selected to plot the relevant Time-Cycle graph. Each sequence in the dataset is represented by a tensor of size <1, 1024, 8>, and the graph demonstrates the values of all 1024 data points collected over a state’s first cycle. Each data point 𝑝𝑝𝑡𝑡 = {𝐹𝐹1𝑡𝑡 … 𝐹𝐹8𝑡𝑡} corresponds to a point in time where the values _(V)_ of all features _{_ 𝐹𝐹1𝑡𝑡 _…_ 𝐹𝐹8𝑡𝑡 _}_ are gathered, while each feature in graph is represented by a different color (figure 4.12). Moreover, it becomes clear from the graphs that the descriptive features are in slightly different scales. However, to simplify data interpretation for the models, normalization is performed here as well through the Robust Scaler ( _scikit-learn_ ), which removes the median and scales the data based on the quantile range, mitigating potential impacts of outliers. 

53 

**Figure 4.11: CSU dataset class balance.** 

**Figure 4.12: Feature scale analysis CSU dataset.** 

54 

To preserve data integrity and avoid the distortion of potential hidden patterns, a decision is made to keep outlier values in both datasets. By preserving outlier values in time series, the models become more resilient to momentary pick measurements which are considered as part of a sequence rather than individual points. The seasonal variations of outlier values can be seen as parts of certain patterns that can trigger the models to distinguish between different machine states. Moreover, since faults are deviations from the normal condition, in many cases models rely on outlier presence to identify the patterns of the anomalies. Finally, the analysis reveals that the time series in both datasets are stationary, meaning that data is changing over time in a rather consistent way, along with their statistical properties, which makes them easier to model. 

## **4.4 Inference Tasks** 

For the experimental part of this study, several DL models with various architectures are implemented to perform two multiclass classification tasks in a supervised learning manner. The first task is performed based on the MaFaulDa dataset, where the goal of the models is to capture the patterns of faults in the sequences of the rotating machine and to demonstrate whether they can detect these faults in unseen data. Similarly, the second task is performed based on the CSU dataset, where the goal of the models is to learn the patterns of faults in the sequences of the gearbox and to detect them in unseen data. Finally, in both tasks, the models should be able to distinguish between the fault and the normal states of the relevant machines. 

|**Inference Task**|**Dataset**|**Machinery Type**|**Number of Models**|
|---|---|---|---|
|Multiclass classification|MaFaulDa|Rotating Machine|9|
|Multiclass classification|CSU|Gearbox|9|



**Table 4.5: Inference tasks.** 

## **4.5 Model Architectures** 

This section describes the architectures of the DL models utilized in the two classification tasks. Each model’s architecture is exported to a graph by using the _Keras plot_model_ module. The exported graphs demonstrate the forward propagation, indicated with arrows, where the inputs are multiplied by the relevant weights in each layer and pass through activation functions to generate a network’s outputs. Each block in the graphs represents a layer in the network, and it contains information regarding the layer’s type and the activation functions, if any, utilized by the neurons in the layer. Additionally, each block contains the layer’s input and output data shapes for both datasets used in the classification tasks, with the shapes for the MaFaulDa dataset appearing on the on the left, and the ones for the CSU on the right. Finally, the output shapes also indicate the corresponding number of neurons in the layer. 

## **4.5.1 Simple RNN** 

This is an implementation of a vanilla RNN model, consisting of an RNN layer with 100 neurons ( _units_ ), using a hyperbolic tangent function ( _activation_ ). To prevent overfitting, the 

55 

RNN layer is configured to perform dropout regularization with probability 0.2 for the linear transformation of the inputs ( _dropout_ ) and the recurrent states ( _recurrent_dropout_ ). An Input layer feeds the RNN layer at each timestep with the values of the parallel series (features) without performing any computations, and a Dense layer, which is fully connected to the RNN layer, is used to generate the model’s outputs by performing the operation demonstrated in (43). 

**==> picture [242 x 13] intentionally omitted <==**

In the output operation, 𝑓𝑓 represents the softmax activation function utilized by the neurons of the output layer, _dot_ denotes the dot product of the layer’s weight vectors with the input vectors, while _bias_ denotes the inductive bias used in the layer. Moreover, the softmax activation function converts the raw output values of the network into probabilities, representing the likelihood of each class. Therefore, the class with the highest probability is considered as the predicted state of the relevant machine. Finally, based on the dataset of each task, the number of neurons in the Dense layer is set to the number of categorical outputs. 

**Figure 4.13: RNN model architecture.** 

## **4.5.2 LSTM** 

The LSTM model consists of an LSTM layer with 100 neurons utilizing a hyperbolic tangent activation function ( _activation_ ) that applies to the cell state and output hidden state, and a sigmoid ( _recurrent_activation_ ) applied to the Input, Forget, and Output gates at each recurrent timestep. To avoid overfitting, the LSTM layer is configured to perform dropout regularization with probability 0.2 for the linear transformation of the inputs and recurrent states. An Input layer feeds the data sequentially to the LSTM layer at each timestep, and a Dense layer with softmax is deeply connected to all neurons in the LSTM layer to produce the classification outputs for the rotating machine and the industrial gearbox. Finally, based on the dataset of each task, the number of neurons in the Dense layer is set to the number of categorical outputs. 

56 

**Figure 4.14: LSTM model architecture.** 

## **4.5.3 GRU** 

The GRU model consists of a GRU layer with 100 neurons, utilizing a hyperbolic tangent activation function that applies to its internal memory, and a sigmoid activation applied to the Update and Reset gates at each timestep. Like the recurrent layers in the previous models, the GRU layer is configured to perform dropout regularization with probability 0.2 for the linear transformation of the inputs and recurrent states. 

**Figure 4.15: GRU model architecture.** 

57 

Finally, an Input layer feeds the time series to the GRU layer at each timestep, and a Dense layer with softmax, fully connected to all neurons in the GRU layer, is used as the network’s output layer to classify each machine’s states. 

## **4.5.4 CNN** 

**Figure 4.16: CNN model architecture.** 

58 

The CNN model consists of an Input layer that provides each sequence to the network as a flat vector. The next layer in the network is a single dimensional convolutional layer (Conv1D) with ReLU activation that applies 100 sliding filters ( _filters_ ) across the input sequences, with the goal to perform hierarchical data decomposition and extract the most relevant features. 

Each filter’s size ( _kernel_size_ ) is set to 8, indicating the window-length of each convolutional operation, where the dot product is computed between the weights of the filter and the values of a sequence within the sliding window. Moreover, each filter moves one step at a time across the time dimension of the input data ( _strides = 1_ ) to ensure that the convolutional operation does not skip any part of the input sequences, thereby extracting important features and preventing information loss. 

As the result of filters sliding across the input sequences, the Conv1D layer generates a feature map that contains the extracted features, which is passed to the next convolutional layer. The second Conv1D layer shares the same specifications, and it is used to extract additional features. The next layer in the network is a max pooling layer (MaxPooling1D) utilized to reduce the complexity and size of feature map, which is provided as input to the layer. 

To achieve this, the MaxPooling1D layer retrieves the maximum value over a sliding window of size 2 ( _pool_size_ ) when applied to the input feature map. The 3D outputs (<sequence length, down-sampled steps, features>) of the MaxPooling1D layer are then passed to a Flatten layer, which converts them into a single-dimensional representation. 

Next, to prevent potential overfitting from continuous convolutions, a Dropout layer is introduced to randomly deactivate some neurons during training, with probability 0.2. Finally, a Dense layer with softmax is utilized to classify the states of the gearbox and the rotating machine, with the number of neurons set to the number of categorical outputs in each dataset. 

## **4.5.5 CNN-LSTM** 

This is a hybrid model combining an LSTM and a CNN network, where the CNN mainly focuses on feature extraction and dimensionality reduction, while the LSTM focuses on capturing temporal dependencies in the input sequences. The CNN processes fixed-length segments of the input sequences and extracts features from each segment, which are then fed to the LSTM to interpret. The input sequences are split based on a fixed-length timestep window and the number of descriptive features in each dataset. Therefore, for the MaFaulDa dataset, each sequence is split into tensors of size <10, 250, 8>, and for the CSU, into tensors of <4, 256, 8>, where each tensor represents <timesteps, fixed-length window, features>. 

This model's CNN network structure mirrors that described in the previous section, with the addition of a Dropout layer with probability 0.2 between the Conv1D layers and the MaxPooling1D layer. The extra regularization layer is used with the intention of minimizing overfitting risks, due to the network’s high complexity and large number of learnable parameters. Additionally, the entire CNN here is wrapped in a Time Distributed layer that allows for the extraction of local features from each window/ segment in the sequence independently, in a way that the temporal ordering of the sequence is still preserved. 

The CNN outputs are then provided to the LSTM layer, which consists of 100 neurons, with the intention of capturing deeper temporal dependencies in the data. The LSTM uses a hyperbolic tangent activation function that applies to the cell state and output hidden state, and a sigmoid activation applied to the Input, Forget, and Output gates at each timestep. 

59 

**Figure 4.17: CNN-LSTM model architecture.** 

Next, a Dropout layer with 0.2 probability is introduced to further prevent overfitting risks, followed by a Dense layer with softmax activation to classify the states of the gearbox and the rotating machine for the relevant classification tasks. 

60 

## **4.5.6 CNN-GRU** 

**Figure 4.18: CNN-GRU model architecture.** 

61 

This is another hybrid model, combining a GRU and a CNN. In this architecture, too, the CNN focuses mainly on local feature extraction and dimensionality reduction, while the GRU aims to capture temporal dependencies in the input sequences more deeply. The model’s architecture is identical to the CNN-LSTM discussed in the previous section, with the only difference being that the LSTM layer is replaced by a GRU layer consisting of 100 neurons. The neurons in the GRU layer utilize a hyperbolic tangent function that applies to the internal memory state, and a sigmoid activation function applied to the Update and Reset gates at each recurrent timestep. 

## **4.5.7 LSTM-Attention** 

The model consists of two main components: an LSTM serving as the encoder and an Attention mechanism. The Attention used in this implementation is demonstrated in Figure 4.20, and it aims to focus on the most relevant segments of the input sequences, in terms of identifying each machine’s states. The LSTM network consists of an LSTM layer with 100 neurons with a hyperbolic tangent activation function applied to the cell state and output hidden state, and a sigmoid activation applied to the Input, Forget and Output gates at each recurrent timestep. 

The LSTM processes the input data sequence and generates outputs at each time step. These outputs are a series of hidden states, where each hidden state is of size 100, across a total number of 2500 time steps. From these outputs, the attention mechanism starts with extracting the last hidden state ℎ𝑡𝑡 (last_hidden_state) using a Lambda layer, which isolates the final state of the LSTM to help focus the Attention mechanism. 

Subsequently, an attention score vector _score_vec_ (attention_score_vec) (44) is computed by applying a Dense layer with a linear activation to each time step's output from the LSTM. This transformation results in a new set of vectors, which are prepared for scoring the relevance of each time step. In the following equation _W_ and _b_ are the weights and bias of the dense layer 

**==> picture [130 x 15] intentionally omitted <==**

The next step involves calculating the actual attention score _score_ (attention_score **)** (45) using a Dot product between the ℎ𝑡𝑡 and the _score_vec_ . This operation computes a score for each time step, indicating the relevance or importance of each step's information. Therefore, this is a multiplicative type of Attention mechanism. 

_score =_ ℎ𝑡𝑡 _⸰_ 𝑠𝑠𝑐𝑐𝑜𝑜𝑎𝑎𝑒𝑒_𝑣𝑣𝑒𝑒𝑐𝑐 (45) 

These scores are then passed through a Softmax layer, which is part of the attention_weight layer. The softmax function converts these attention scores into a probability distribution 𝑃𝑃𝑡𝑡 (46), indicating the relative importance of each time step's data in the form of weights. 

**==> picture [124 x 15] intentionally omitted <==**

The weighted sum of the original LSTM outputs, weighted by the attention weights 𝑤𝑤𝑡𝑡, is computed next. This is done through another Dot product, combining the attention weights (attention_weight) and the LSTM outputs to produce the context vector 𝑐𝑐𝑡𝑡 (contect_vector). The 𝑐𝑐𝑡𝑡 summarizes the most relevant information from the entire sequence as dictated by the attention mechanism. Following the creation of the 𝑐𝑐𝑡𝑡, the attention mechanism proceeds to generate the attention output (attention_output **)** . This is achieved by concatenating the ℎ𝑡𝑡with the 𝑐𝑐𝑡𝑡, which integrates both the immediate last output and the contextual information deemed relevant by the attention weights. This concatenated output is then passed through a Dense layer 

62 

with a hyperbolic tangent activation to produce an attention vector 𝑑𝑑𝑡𝑡 (attention_vector) (47), which refines the data into a more manageable form for final output processing. 

**==> picture [125 x 15] intentionally omitted <==**

Finally, the output of the Attention is provided to a Dense layer used as the output layer of the LSTM-Attention model to generate the classification outputs. The Dense layer utilizes a softmax activation, and similarly to the previous models, the number of neurons in the layer is set to the number of categorical outputs in each dataset. The detailed architecture of the Attention layer is demonstrated in figure 4.20 with the shapes of the MaFaulDa dataset. Moreover, to enhance clarity, the graph includes aliases for each layer, as described above. 

**Figure 4.19: LSTM-Attention model architecture.** 

63 

**Figure 4.20: Architecture of the Attention layer in LSTM-Attention model.** 

## **4.5.8 GRU-Attention** 

The model’s architecture is identical to that of the LSTM-Attention discussed in the previous section, along with the structure of the Attention mechanism. The only difference here is that the LSTM layer is replaced by a GRU with 100 neurons acting as the encoder. The GRU utilizes a hyperbolic tangent activation function applied to the internal memory state, and a sigmoid activation function applied to the Update and Reset gates at each recurrent timestep. 

64 

**Figure 4.21: GRU-Attention model architecture.** 

## **4.5.9 Transformer** 

The Transformer model operates with an encoder architecture tailored for classification, thus omitting the decoder typical of sequence-to-sequence tasks. This encoder comprises two encoder blocks, each consisting of a multi-head attention mechanism and a feed-forward network. 

## **Multi-Head Attention Mechanism:** 

Each block begins with a layer normalization (LayerNormalization), which normalizes the inputs to have zero mean and unit variance, aiding the model in learning effectively across deep network structures. Following this, a multi-head attention mechanism (MultiHeadAttention) processes the inputs. In this typical setup, each of the multiple heads (e.g., two heads) operates independently, with each head having its own query (Q) and key (K) vectors. The dimensionality of these vectors in each head is set to 8 ( _key_dim_ ). 

This configuration means that while each head processes information separately, the combined capacity of the heads enriches the model's ability to discern complex patterns in data without summing the dimensions across heads. To mitigate the risk of overfitting during training, a dropout rate of 0.2 is applied within the multi-head attention layer, ensuring robustness in the learned attention patterns. The outputs from these individual heads are integrated through a process that combines their contributions, preserving both the original input and transformed features via a residual connection executed by a subsequent lambda transformation (TFOpLambda). 

65 

**Figure 4.22: Transformer model architecture.** 

66 

## **Feed-Forward Network:** 

Subsequent to the attention, the feed-forward block employs another layer normalization, followed by a dense layer with ReLU activation to introduce non-linear processing capabilities. This section also incorporates a residual connection through a transformation operation (TFOpLambda), enabling the preservation of the input's integrity while facilitating feature transformations. 

## **Classification Preparation:** 

To transition to classification, the GlobalAveragePooling1D layer condenses the encoder output, reducing its spatial dimensions by averaging over the sequence length to produce a fixed-length output vector. 

## **Classifier Network:** 

This vector feeds into the classifier comprising a dense layer with 50 ReLU-activated neurons, a dropout layer with a specified dropout rate of 0.2 to mitigate overfitting, and finally, a dense layer employing softmax activation to generate the classification outputs according to the task's requirements. 

## **4.6 Learning Process** 

## **Data Splitting:** 

Each classification task's learning and inference process necessitates a specific data splitting strategy. Here, for both datasets, 70% of the data is used for training, 10% for validation, and the remaining 20% for testing. The models use the training set to identify fault patterns in a supervised learning context, while the validation set helps the models to generalize better by exposing them to unseen data, thus preventing overfitting. The validation set serves to offer an unbiased assessment model performance, so that the best combination of its hyperparameters can be selected for the inference phase. In the final phase, the testing set is used to assess whether the models can detect fault in the unseen data. Finally, the data is split by using the _train_test_split_ module of the _scikit-learn_ library. 

**Figure 4.23: Dataset split method.** 

67 

|**Set**|**Number of Sequences**|**Tensor Representation**|
|---|---|---|
|Training|1404|<1404, 2500, 8>|
|Validation|159|<159, 2500, 8>|
|Testing|391|<391, 2500, 8>|



**Table 4.6: MaFaulDa dataset splitting approach.** 

|**Set**|**Number of Sequences**|**Tensor Representation**|
|---|---|---|
|Training|3680|<3680, 1024, 8>|
|Validation|410|<410, 1024, 8>|
|Testing|1025|<1025, 1024, 8>|



**Table 4.7: CSU dataset splitting approach.** 

## **Parametrization:** 

For the supervised learning process in both tasks, the Adam optimizer is used, a type of stochastic gradient descent algorithm, which is responsible for finding the global minimum and update a model’s trainable weights (table 4.8) through backpropagation. 

|**Model**|**Trainable**|**Parameters**|
|---|---|---|
||**MaFaulDa**|**CSU**|
|RNN|11506|11405|
|LSTM|44206|44105|
|GRU|33606|33505|
|CNN|832406|339105|
|CNN-LSTM|4847606|4967505|
|CNN-GRU|3657806|3747705|
|LSTM-Attention|63906|63855|
|GRU-Attention|53306|53255|
|Transformer|50752|50501|



**Table 4.8: Number of trainable parameters based on the dataset for each Keras model.** 

Furthermore, the learning process uses a probabilistic loss function to evaluate deviations between actual and target outputs, guiding model adjustments based on the loss score. For both classification tasks, the sparce categorical cross entropy function is selected, which utilizes a logarithmic-based penalization mechanism, where higher penalties are applied to greater deviations. The function can be formulated as shown in (48), where 𝑙𝑙𝑖𝑖 denotes a sample’s target class, 𝑝𝑝𝑖𝑖 the generated softmax probability for the sample to belong to class _i_ , and _n_ is the number of classes in the dataset. 

**==> picture [151 x 23] intentionally omitted <==**

To enhance learning accuracy, albeit at the cost of longer training durations, a fixed learning rate of 0.001 is maintained throughout the training process. The intention of this decision is to minimize the probability of the optimizer settling for suboptimal weights. Additionally, to 

68 

optimize model performance and minimize overfitting risks, a callback policy with checkpoints is adopted, where only the weights of the best training model are saved after each iteration, based on the improvement of the validation loss score. This approach ensures that the optimal set of weights, determined during training, is restored for use during the inference process, maximizing model effectiveness. 

Moreover, to prevent the gradients from vanishing or exploding, the Glorot weight initialization approach is adopted for all models. This approach initializes the weights based on a uniform distribution within _[-N, N]_ (49), with _I_ and _O_ denoting the number of input and output units in each layer respectively, while the values of the initial bias vector in all models are set to 0. 

**==> picture [118 x 29] intentionally omitted <==**

Finally, the Batch Gradient Descent approach is adopted, where the error is computed for each input sequence individually, and then the mean gradient of all sequences in the training set is used at each iteration to update the trainable parameters. 

|**Parameter**|**Value**|
|---|---|
|optimizer|adam|
|loss|sparce_categorical_crossentropy|
|metric|accuracy|
|monitor|val_loss|
|learning_rate|0.001|
|kernel_initializer|glorot_uniform|
|bias_initializer|zeros|



**Table 4.9: Training parametrization of the Keras models.** 

69 

## **5. Evaluation** 

## **5.1 Rotating Machine Fault Detection** 

The classification task for the rotating machine is based on the MaFaulDa dataset, where each sequence in the training and test sets consists of 2,500 data points with 8 parallel features, as described in table 4.2. For the supervised learning process, a label has been assigned to every sequence with values from 0 to 5, representing its ground-truth class. Moreover, the models are trained in different numbers of epochs, where each epoch corresponds to a full pass of the entire training set. Each model’s training stops when its learning curve cannot be further improved based on its validation accuracy history. For the inference phase, 391 sequences are used in the test set to evaluate model performance. The test set reflects the complete dataset’s class balance, as demonstrated in figure 5.1. 

**Figure 5.1: MaFaulDa test set class distribution.** 

The evaluation of each model’s performance is based on the accuracy score, average macro F1 score, average macro precision, and average macro recall. The macro average scores are calculated by averaging the metric scores across all classes in the dataset. Using the macro averaging approach ensures that all classes contribute equally to the model’s learning process, regardless of class imbalance. Finally, each model’s scores in the inference phase of the task are illustrated in the table 5.1, with the best scores highlighted in bold. 

70 

|**DL** **Model**|**Accuracy**<br>**(%)**|**Macro Avg**<br>**F-1 Score**|**Macro Avg**<br>**Precision**|**Macro Avg**<br>**Recall**|**Number of**<br>**Epochs**|
|---|---|---|---|---|---|
|RNN|91.56|0.794|0.833|0.785|150|
|LSTM|97.70|0.957|0.944|0.976|150|
|GRU|97.95|0.916|0.940|0.901|150|
|CNN|97.44|0.973|0.970|0.977|50|
|CNN-LSTM|98.98|0.991|0.992|0.991|60|
|CNN-GRU|99.23|0.993|0.994|0.993|60|
|LSTM-Attention|99.23|0.991|0.991|0.992|100|
|GRU-Attention|99.49|0.996|0.997|0.996|100|
|Transformer|**99.74**|**0.998**|**0.998**|**0.998**|**50**|



**Table 5.1: Inference phase performance with the MaFaulDa dataset.** 

The performance of the models is further analyzed by utilizing the confusion matrix, which provides an analytical overview of how the sequences of each state are being classified. The confusion matrix indicates how accurately the models identify each state’s patterns and to what extent they can distinguish between them. The results of the confusion matrix for each model are summarized in the following figures, where the number of correctly classified sequences per state appears in the diagonal positions of the matrix, while the rest of the positions represent the number of misclassified sequences. Finally, In the matrix, the x-axis represents model predictions, and the y-axis shows actual labels. 

**Figure 5.2: RNN and LSTM confusion matrices on the MaFaulDa dataset.** 

71 

**Figure 5.3: GRU and CNN confusion matrices on the MaFaulDa dataset.** 

**Figure 5.4: CNN-LSTM and CNN-GRU confusion matrices on the MaFaulDa dataset.** 

72 

**Figure 5.5: LSTM-Attention, GRU-Attention, and Transformer confusion matrices on the MaFaulDa dataset.** 

The results of the confusion matrices are further summarized in table 5.2, where the numbers of correctly classified and misclassified sequences per model are shown. Moreover, the Captured Faults column refers to the number of correctly identified fault states in the test dataset, while the Missed Faults column corresponds to the cases where a fault state is misclassified as a normal condition, and therefore no counteraction can be put into effect. Finally, the False Alarm column signifies the number of times a model misclassifies a normal condition of the rotating machine as faulty, and as a result, a false alarm is raised. 

73 

|**DL** **Model**|**Correctly**<br>**Classified**<br>**Sequences**|**Misclassified**<br>**Sequences**|**Captured**<br>**Faults**|**Missed**<br>**Faults**|**False**<br>**Alarms**|
|---|---|---|---|---|---|
|RNN|368|23|357|1|5|
|LSTM|382|9|376|1|0|
|GRU|383|8|380|1|3|
|CNN|381|10|375|0|0|
|CNN-LSTM|387|4|381|0|0|
|CNN-GRU|388|3|382|0|0|
|LSTM-Attention|388|3|382|0|0|
|GRU-Attention|389|2|383|0|0|
|Transformer|**390**|**1**|**384**|**0**|**0**|



**Table 5.2: Summary of the confusion matrix results with the MaFaulDa dataset.** 

The performance of the inference models is further investigated by analyzing their ability to distinguish between the different states of the rotating machine using the ROC-AUC metric. Since the ROC-AUC metric typically applies to binary classification tasks, for the multiclass classification tasks in this study, the One-Versus-Rest (OvR) approach is adopted. This approach evaluates how well each model can differentiate between the sequences in each dataset. Specifically, with the OvR approach, the multiclass classification problem is split into several binary classification problems, where each evaluated class is considered the positive class, and the rest of the classes aggregated are considered the negative class. The following figures demonstrate the ROC-AUC graphs for all inference models. 

**Figure 5.6: ROC-AUC graph of the RNN on the MaFaulDa dataset.** 

74 

**Figure 5.7: ROC-AUC graphs of the LSTM, GRU, and CNN on the MaFaulDa dataset.** 

75 

**Figure 5.8: ROC-AUC graphs of the CNN-LSTM, CNN-GRU, and LSTM-Attention on the MaFaulDa dataset.** 

76 

**Figure 5.9: ROC-AUC graphs of the GRU-Attention and Transformer on the MaFaulDa dataset.** 

To verify the results of the inference models and gain insights on their learning process, the training and validation history of the categorical cross-entropy loss function is reviewed for each model, along with the history of the accuracy score. Analyzing these graphs helps draw several conclusions about the models' learning processes. More precisely, it can be concluded whether the validation set is too small for the given training set, or the reverse, and therefore the data split method must be reconsidered. Furthermore, it can be determined whether the validation set is easier to predict than the training set, which could explain lower performance results with the unseen data. 

Another observation that can be made is whether a model overfits, underfits, or fits the training data appropriately. These graphs also help determine how quickly each model converges and whether the number of training epochs is adequate or if further improvement is needed. The following graphs illustrate each model’s loss and accuracy history with the MaFaulDa dataset, where the blue line represents training accuracy and loss, and the orange line validation accuracy and loss. 

77 

**Figure 5.10: Accuracy and loss history of the RNN on the MaFaulDa dataset.** 

**Figure 5.11: Accuracy and loss history of the LSTM on the MaFaulDa dataset.** 

78 

**Figure 5.12: Accuracy and loss history of the GRU on the MaFaulDa dataset.** 

**Figure 5.13: Accuracy and loss history of the CNN on the MaFaulDa dataset.** 

79 

**Figure 5.14: Accuracy and loss history of the CNN-LSTM on the MaFaulDa dataset.** 

**Figure 5.15: Accuracy and loss history of the CNN-GRU on the MaFaulDa dataset.** 

80 

**Figure 5.16: Accuracy and loss history of the LSTM-Attention on the MaFaulDa dataset.** 

**Figure 5.17: Accuracy and loss history of the GRU-Attention on the MaFaulDa dataset.** 

81 

**Figure 5.18: Accuracy and loss history of the Transformer on the MaFaulDa dataset.** 

To gain insights on each model’s training and inference times with the MaFaulDa dataset, the relevant graphs are plotted with time recorded in seconds. 

**Figure 5.19: Model training times with the MaFaulDa dataset.** 

82 

**Figure 5.20: Model inference times with the MaFaulDa dataset.** 

## **5.2 Industrial Gearbox Fault Detection** 

The classification task for fault detection in the industrial gearbox is based on the CSU dataset, with each sequence in the training and test sets consisting of 1,024 data points with 8 parallel features, as described in table 4.4. 

**Figure 5.21: CSU test set class distribution.** 

83 

For the supervised learning process, each sequence has been associated with a label, with values from 0 to 4, representing the sequence’s ground-truth class. Similarly to the previous task, the models are trained in varying epochs and stop when no further improvements in learning or accuracy are seen. For the inference phase, 205 sequences from each state are randomly selected for the test set (Figure 5.21), and the same evaluation metrics are used for this task as well. Table 5.3 displays the inference performance results for each model, highlighting the highest scores in bold. 

|**DL Model**|**Accuracy**<br>**(%)**|**Macro Avg**<br>**F-1 Score**|**Macro Avg**<br>**Precision**|**Macro Avg**<br>**Recall**|**Number of**<br>**Epochs**|
|---|---|---|---|---|---|
|RNN|93.27|0.933|0.933|0.933|100|
|LSTM|97.56|0.976|0.976|0.976|100|
|GRU|98.05|0.981|0.980|0.981|100|
|CNN|97.76|0.978|0.978|0.978|100|
|CNN-LSTM|99.12|0.991|0.991|0.991|100|
|CNN-GRU|99.22|0.992|0.992|0.992|100|
|LSTM-Attention|99.51|0.995|0.995|0.995|150|
|GRU-Attention|99.61|0.996|0.996|0.996|150|
|Transformer|**99.90**|**0.999**|**0.999**|**0.999**|**50**|



**Table 5.3: Inference phase performance with the CSU dataset.** 

Like the previous task, the performance of the models is analyzed further based on the confusion matrix which provides an analytical overview on how the sequences of each gearbox state are being classified. The results of the confusion matrices are further summarized in the table 5.4. 

**Figure 5.22: RNN and LSTM confusion matrices on the CSU dataset.** 

84 

**Figure 5.23: GRU and CNN confusion matrices on the CSU dataset.** 

**Figure 5.24: CNN-LSTM and CNN-GRU confusion matrices on the CSU dataset.** 

85 

**Figure 5.25: LSTM-Attention, GRU-Attention, and Transformer confusion matrices on the CSU dataset.** 

86 

|**DL Model**|**Correctly**<br>**Classified**<br>**Sequences**|**Misclassified**<br>**Sequences**|**Captured**<br>**Faults**|**Missed**<br>**Faults**|**False**<br>**Alarms**|
|---|---|---|---|---|---|
|RNN|956|69|766|13|15|
|LSTM|1000|25|799|10|4|
|GRU|1005|20|800|0|0|
|CNN|1002|23|798|0|1|
|CNN-LSTM|1016|9|811|3|0|
|CNN-GRU|1017|8|812|0|0|
|LSTM-Attention|1020|5|816|0|1|
|GRU-Attention|1021|4|816|0|0|
|Transformer|**1024**|**1**|**819**|**0**|**0**|



**Table 5.4: Summary of the confusion matrix results with the CSU dataset.** 

The performance of the inference models in this task is also investigated deeper by utilizing the ROC-AUC with the OvR approach, to assess each model’s level of separability with the gearbox states. The following graphs demonstrate each model’s ROC-AUC scores. 

**Figure 5.26: ROC-AUC graph of the RNN on the CSU dataset.** 

87 

**Figure 5.27: ROC-AUC graphs of the LSTM, GRU, and CNN on the CSU dataset.** 

88 

**Figure 5.28: ROC-AUC graphs of the CNN-LSTM, CNN-GRU, and LSTM-Attention on the CSU dataset.** 

89 

**Figure 5.29: ROC-AUC graphs of the GRU-Attention and Transformer on the CSU dataset.** 

To better understand the models' learning processes and validate their performance with the CSU dataset, we analyze the training and validation histories of both loss and accuracy. The graphs below show each model’s loss and accuracy over time, with the blue line representing training metrics and the orange line representing validation metrics. 

90 

**Figure 5.30: Accuracy and loss history of the RNN on the CSU dataset.** 

**Figure 5.31: Accuracy and loss history of the LSTM on the CSU dataset.** 

91 

**Figure 5.32: Accuracy and loss history of the GRU on the CSU dataset.** 

**Figure 5.33: Accuracy and loss history of the CNN on the CSU dataset.** 

92 

**Figure 5.34: Accuracy and loss history of the CNN-LSTM on the CSU dataset.** 

**Figure 5.35: Accuracy and loss history of the CNN-GRU on the CSU dataset.** 

93 

**Figure 5.36: Accuracy and loss history of the LTM-Attention on the CSU dataset.** 

**Figure 5.37 Accuracy and loss history of the GRU-Attention on the CSU dataset.** 

94 

**Figure 5.38: Accuracy and loss history of the Transformer on the CSU dataset.** 

Finally, to gain insights regarding each model’s training and inference times with the CSU dataset, the relevant graphs are plotted with time recorded in seconds. 

**Figure 5.39: Model training times with the CSU dataset.** 

95 

**Figure 5.40: Model inference times with the CSU dataset.** 

96 

## **6. Discussion** 

## **6.1 Classification Tasks** 

The evaluation results of the experiments conducted in this study show that the DL models achieved high performance in detecting the fault states of the rotating machine and the industrial gearbox. More precisely, based on the tables 5.1 and 5.3, eight out of nine models obtained an accuracy score higher than 97% in both classification tasks. 

The Transformer model obtained the best classification accuracy in both tasks, reaching a score of 99.74% with the MaFaulDa dataset and 99.90% with the CSU. Models that combined the Attention mechanism or CNN with advanced recurrent networks also achieved particularly high classification accuracy. More specifically, the accuracy of the Attention-based models, such as the LSTM-Attention and GRU-Attention, ranged between 99.23% and 99.61% in both classification tasks, while the CNN-based models, like the CNN-LSTM and CNN-GRU, obtained scores in the range of 98.98% to 99.23%. 

Single-network models, like the CNN and the advanced recurrent networks such as LSTM and GRU, also demonstrated strong accuracy performance in both tasks, yet at the lower range of 97.44% to 98.05%. It is worth noting that the accuracy scores achieved by the GRU and CNN on the MaFaulDa dataset are comparable to those reported in previous studies [29, 41] that used similar models with the dataset. Likewise, the LSTM’s performance on the CSU was found roughly equal to that reported by [38] on the same dataset. However, the RNN’s accuracy was notably lower compared to the other models, reaching an accuracy score of 91.56% with the MaFaulDa dataset and 93.27% with the CSU. 

A general observation that can be made from the relevant tables is that the models’ average classification accuracy with the CSU dataset was slightly higher compared to the one with the MaFaulDa dataset, which could be attributed to the lengthier sequences of the latter. 

Moreover, by observing the obtained precision scores, it becomes apparent that the DL models were, in general, quite precise in detecting the states of the rotating machine and the industrial gearbox, while the relevant recall scores reveal that they could discriminate the states with high sensitivity. Likewise, the achieved F-1 scores showed that the models in general could successfully identify each state, while effectively discriminating them from the other states. The F-1 score is particularly useful for interpreting the classifiers’ performance with the MaFaulDa dataset, given that is imbalanced. 

Nevertheless, the DL networks were resilient to this extent of class imbalance, with deeper models such as the Transformer, the Attention-based LSTM and GRU, and the hybrid CNNLSTM and CNN-GRU achieving an F-1 score higher than 0.991. Single-network models also coped well with the MaFaulDa dataset, where the CNN and LSTM obtained F-1 scores of 0.973 and 0.957, respectively, while the GRU demonstrated a slightly less satisfactory score of 0.916. In line with the accuracy scores, the Transformer achieved the highest F-1 score (0.998), while the RNN’s score was significantly lower (0.794) compared to the other models. 

Furthermore, the good results of the models in both tasks can be further verified by analyzing the confusion matrices, where it becomes apparent that the models were able to correctly capture patterns of various fault types in the rotating machine and the industrial gearbox during their learning process. Additionally, they were able to successfully detect those patterns in the unseen sequences and, at the same time, discriminate them from the normal conditions. Especially by observing the confusion matrices for the MaFaulDa dataset, it becomes evident 

97 

that the models do not particularly misclassify minor states compared to the major ones. However, the RNN severely misclassified the normal states of the rotating machine, which is the most minor state in terms of class distribution. 

The results of the confusion matrices for the MaFaulDa and CSU datasets are summarized in tables 5.2 and 5.4, respectively. The tables show that the models were able to capture significant portions of faults in the rotating machine and the industrial gearbox, with the Transformer model demonstrating the highest fault detection rates in both tasks. Other models, such as the Attention-based and hybrid ones, were also capable of capturing the vast majority of faults in both datasets. These were followed by the single-network models, and finally the RNN, which captured the smallest share of faults. 

While identifying faults is the main objective of such models in industrial contexts, reducing the number of false alarms can be of equivalent importance due to the associated costs. In both tasks, the DL networks illustrated their capabilities in minimizing false alarms, with seven out of nine models producing zero false alarms when processing the states of the rotating machine, while five models did not generate any false alarms for the gearbox states. 

Another metric utilized to evaluate the performance of the explored solutions is the ROC-AUC. By analyzing the corresponding graphs, it can be seen that, in general, the models were able to distinguish between the states of the relevant equipment with high confidence. It is remarkable that models like the Transformer, Attention-based LSTM and GRU, and the hybrid models obtained an AUC score of 1 with all states in both classification tasks. This shows the networks' effectiveness in deeply capturing each state’s distinct patterns during the learning phase, allowing them to detect those patterns in the unseen sequences. 

The learning curves of the loss function and accuracy, depicting the models’ training and validation progress, provide further evidence of their potent learning capabilities. More specifically, from the corresponding graphs, it becomes clear that all models can learn to detect the states of the rotating machine and the industrial gearbox, with almost all models demonstrating a rather smooth and stable learning process. However, the RNN exhibited slight instability towards the end of its learning process with the MaFaulDa dataset. 

Another observation that can be made from the training-validation history graphs is that the models do not overfit or underfit the training sequences. For all models, the training and validation curves are well aligned, indicating comparable performance without prominent deviations. In both training and validation, while the accuracy improves with the number of iterations, the loss steadily leads to a minimum point, reflecting good generalization, as verified by the models’ performance with the unseen sequences. 

From the same graphs, it also becomes apparent that some models were able to converge faster than others, requiring fewer training iterations, which is related to their internal mechanisms and approaches to process the input sequences during training. Similarly, by observing the training time graphs, it becomes clear that each model’s training time is correlated with its convergence rate, where models that converged faster typically required less training time. 

Finally, all models achieved similar inference times on the test sequences, ranging approximately between 4 to 6.5 seconds in both tasks. Training is computationally intensive, requiring repeated adjustments of model parameters, backpropagation, and multiple passes through the training dataset to deeply capture fault patterns. However, once trained, inference simply involves applying the learned parameters to new data, a much faster process where the input data is processed only once. 

98 

## **6.2 Reflection on the Research Questions** 

While the previous section summarized evaluation outcomes and general observations from the classification tasks, this section discusses how the experimental results address the research questions listed in Section 1.5. 

## **6.2.1 Research Question 1** 

To address the problem of fault detection in industrial machinery, DL models of various architectures were utilized to detect fault states in a rotating machine and an industrial gearbox. Understanding the architecture of DL networks is crucial for successful fault detection. Thus, in the modeling part of this study, the explored networks were developed in Python programming language, using the Keras framework, which allowed for generating graphical representations of the models' architectures. By doing so, a clear and concise representation of each model's structure is provided, which can help in better understanding their underlying mechanisms and evaluating their effectiveness in detecting the fault states. 

More specifically, presenting a visual overview of each model’s architecture, including the input and output shapes at each layer, facilitated the comprehension of the information flow and processing steps within the network. Additionally, by examining the layers and connections between them, the process of identifying the key features leveraged by each model to detect the fault states can be facilitated. Similarly, by analyzing each layer’s activation functions, a deeper understanding of how each network regulates traffic can be achieved. 

While Keras provides a graphical representation of each model’s structure, it may only offer a high-level abstraction of the displayed architectures. Therefore, to achieve a more granular understanding, the study included a detailed theoretical description in the Theory section. The description delves into the inner workings of each model's architecture and the DL concepts utilized in layers like the multi-head Attention layer in the Transformer model, recurrent layers such as the RNN, LSTM, and GRU, and the Convolutional and Pooling layers used in the CNN. 

In addition to providing the architectures, a detailed configuration of the networks’ training process was presented, along with the necessary metrics to evaluate their performance. This ensures that the models were trained and evaluated under consistent and well-defined conditions, which is essential for reproducibility and comparability of results and critical for their application in real-world fault detection scenarios. The training details included information on the input data format, the preprocessing steps, the hyperparameters of the models, and the optimization algorithms used in the learning process. 

This level of interpretability is important for real-world applications, where it is essential to understand how DL models can be utilized to detect faults in industrial machines, and which factors may contribute to their performance. Finally, the architectures provided can serve as a basis for future research, contributing to the development of more effective DL solutions for the domain. Therefore, this information is shared with the hope of facilitating the replication and extension of this work by other researchers in the field. 

99 

## **6.2.2 Research Question 2** 

To further evaluate the suitability of DL models for this domain, a comparative analysis was conducted with previous studies that employed traditional ML and shallow learning techniques on the same datasets. In the task of detecting faults in the rotating machine with the MaFaulDa dataset, eight out of nine DL models outperformed the traditional ML algorithms presented in table 6.1, which were previously applied to the same dataset. A similar outcome was observed in the task of detecting faults in the industrial gearbox. Specifically, as illustrated in table 6.2, eight out of the nine explored DL solutions achieved better performance compared to some traditional ML models from previous studies, when applied to the CSU dataset. 

|**Reference**|**Explored Models**|**Preprocessing Techniques**|**Classification**<br>**Accuracy (%)**|
|---|---|---|---|
|[30]|Logistic Regression,<br>CatBoost|- Dimensionality reduction with FFT<br>-Feature extraction with TSFEL algorithm|CB: 90.00<br>LR: 93.00|
|[31]|Random Forest|- Feature generation (kurtosis, entropy)<br>-Feature extraction with SBM|94.14|
|[32]|MLP|- Dimensionality reduction with FFT<br>- Class balance by oversampling with zero-<br>mean Gaussian noise|93.50|
|[33]|MLP|- Class balance by oversampling with<br>Gaussian white noise at a rate of 10db-SNR|95.80|
|[34]|MLP|-Class balance with SMOTE oversampling|96.20|
|[35]|MLP|- Class balance with SMOTE oversampling<br>-Feature extraction with Relief algorithm|96.93|



**Table 6.1: An overview of traditional ML models in studies using the MaFaulDa dataset.** 

|**Reference**|**Explored Models**|**Preprocessing Techniques**|**Classification**<br>**Accuracy (%)**|
|---|---|---|---|
|[38]|SVM|- Data denoising with EMD|88.16|
|[36]|SVM, KNN, MLP|- Feature extraction with Wavelet<br>- Time-windowing for representation<br>downsizing|MLP: 90.06<br>KNN: 93.20<br>SVM: 94.40|
|[37]|Random Forest|- Data denoising and feature extraction<br>with SVDS<br>-Bootstrap resampling|0% D.O.: 90.90<br>90% D.O.: 96.30|



**Table 6.2: An overview of traditional ML models in studies using the CSU dataset.** 

Industrial machinery data is typically collected in the form of time series, reflecting how fault patterns evolve over time. One limitation of most traditional ML models, such as the ones presented in tables 6.1 and 6.2, is that they require the input time series to be decomposed into individual data points of tabular representation by utilizing windowing techniques. This might cause important information arising from time dependencies and the order of observations to be lost in the decomposition process, especially when long-range relationships exist between observations. 

100 

To preserve such contextual information, the input sequences were provided in full to the explored DL models during their training process, which was feasible due to their ability to process data sequentially or in parallel, depending on the DL technology. Sequential data processing, performed by recurrent networks, considers temporal dependencies between observations based on information from previous timesteps. This enables the models to capture hidden patterns and trends in the states of the rotating machine and the industrial gearbox that might be indicative of specific fault types. 

Similarly, parallel processing performed by the CNN and the Transformer enables the models to identify the most important information from multiple segments of the input sequences simultaneously, which can also lead to capturing time-localized patterns that might be unique for each state. These advanced approaches can be more effective than analyzing individual data points which may not reveal such patterns, enhancing a fault detector’s accuracy during condition monitoring. 

The comparison also highlighted that DL networks can achieve high fault detection performance while utilizing simpler and less elaborate data preprocessing techniques. The data preprocessing pipeline for detecting faults in the rotating machine and the industrial gearbox did not involve any manual feature extraction techniques. In contrast to some traditional ML techniques that employed additional feature extraction algorithms [30, 31, 35, 36], or derived new statistical features to capture important information in the data before providing it to the models [31], the explored DL networks were able to effectively learn the relevant features directly from the normalized data. 

In three of the explored DL architectures, including the CNN, CNN-LSTM, and CNN-GRU, the CNN was utilized to automatically extract time-localized features by applying convolutional and pooling operations to the input sequences over fixed-length filters. Additionally, the Attention mechanism was utilized in the LSTM-Attention, GRU-Attention, and Transformer models to focus more on the most informative parts of the input sequences in terms of detecting the target machine state. As observed from the classification results of the relevant models, both the CNN and Attention mechanism proved effective in identifying the states of the rotating machine and the industrial gearbox by selectively focusing on relevant information. 

Utilizing CNNs for automatic feature extraction and the Attention mechanism to highlight the most important parts of the input sequences can also improve robustness to noise. While some previous studies [37, 38] applied signal denoising techniques to the CSU dataset before providing the data to the ML models, in this study the normalized sequences were given directly to the DL networks during training. By focusing only on the most relevant information for identifying fault states, DL models can ignore noisy or irrelevant parts in the input signals and achieve good generalization without fitting the noisy data. 

The risks of overfitting noisy or redundant data from the input sequences can also be minimized by incorporating regularization techniques, which are built into the DL technologies. In the implementation phase, dropout layers were employed to randomly drop some neurons during training, which prevents a model from focusing too much on neurons that might be introducing noise to the network. The good generalization of the DL models is evidenced by their classification accuracy with the unseen sequences, and the absence of overfitting indications in the accuracy-loss history graphs. 

It is important to note, though, that while the amount of noise in the CSU dataset might not be significant enough to substantially affect the overall performance of the DL models, this does not imply that they are immune to every type of noise. Time series data from industrial machines can contain noise at different scales, which might impact the performance of DL models. 

101 

Moreover, the amount of noise in the data can vary depending on the nature of the industrial equipment and the data acquisition methods. Notably, by utilizing the Robust Scaler during normalization, the sequences of the CSU dataset were scaled based on their quantile range, which increased the models' robustness to outliers, as a form of noise in the data. 

Furthermore, when it comes to MaFaulDa dataset, the dimensionality of the sequences was downsized by resampling at the rate of 1 kHz, which based on the FFT analysis was the band that contained high signal strength frequencies. This way, bands with less significant frequency components and relatively less information in their range were excluded from the resampling, which can potentially reduce the amount of noise in the data. This technique has been utilized in other studies [29, 30, 32, 41] as well, applying both DL or traditional ML techniques on the dataset, and it can facilitate the models to demonstrate better robustness to noise. 

Additionally, in some studies [32, 33, 34, 35] where ML models were applied to the MaFaulDa dataset, oversampling methods were employed during the data preprocessing phase to minimize the impact of class imbalance. Although oversampling techniques can be effective in mitigating class imbalance, the quality of the acquired synthetic data can notably influence the performance of the classifiers. 

Consistent with DL models from prior research [29, 41] applied to the dataset, the models examined in this study were able to classify the states of the rotating machine with high accuracy, despite the class imbalance. While both DL and ML models can be substantially affected by class imbalance, one possible explanation is that the models were trained to classify sequences rather than individual data points. By feeding the models entire sequences during training, contextual information such as local patterns, trends, and time dependencies between observations can be considered. This approach may give to the DL models the benefit of making classification decisions based on more prominent information compared to when classifying individual data points. 

Architectures based on recurrent networks utilize the concept of memory to identify the states of the rotating machine and the industrial gearbox by considering information from previous timesteps to make classification decisions at the current time. Similarly, CNN-based models perform automatic feature extraction to effectively capture important information in the input sequences. In the same manner, the Attention mechanism used in the LSTM-Attention, GRUAttention, and Transformer models highlights the most relevant information in the input sequences. 

This information tends to contain the most discriminative features, which might be unique for each state of the rotating machine, regardless of class distribution. As a result, these models can more easily distinguish between the machine’s different states during the inference phase, without being significantly affected by class imbalance. This is also evident in the confusion matrices from the first task, where minor states in the MaFaulDa dataset were not particularly misclassified compared to the more frequent states. 

Robustness to class imbalance is particularly beneficial for fault detection tasks in industrial machines, where some fault types may be rare compared to others. Consequently, models may be trained on imbalanced datasets. On the other hand, this does not suggest that DL models can be immune to every level of class imbalance. DL models, while powerful, can be data-intensive and may exhibit diminished performance due to class imbalance, which often varies significantly across different types of tasks and settings. 

102 

## **6.2.3 Research Question 3** 

As observed from the evaluation metrics, all DL models obtained high accuracy rates in both classification tasks, demonstrating effectiveness in learning different fault patterns. Despite the relatively small variability in their performance though, some models can be considered more successful in detecting the fault states of the rotating machine and the industrial gearbox. 

When it comes to the recurrent networks, the LSTM and GRU demonstrated superior fault detection capabilities compared to the basic RNN. This could be explained due to the LSTM’s and GRU’s more sophisticated recurrent structures which allow them to classify the states of the rotating machine and the industrial gearbox based on information from longer periods of time. The LSTM and GRU employ various gates to optimally regulate information flow, effectively addressing the vanishing gradient problem in the long sequences of the MaFaulDa and CSU datasets. This challenge made the RNN more susceptible to the length of input sequences in each task, affecting its performance more than the other models. Additionally, the RNN showed the highest sensitivity to class imbalance, detecting only one sequence from the normal state of the rotating machine, the least frequent class. 

In line with other studies in the field [2, 22, 23] that utilized recurrent networks to address the problem in different types of industrial equipment, the GRU outperformed the LSTM in capturing portions of faults in the rotating machine and the industrial gearbox. The GRU was the most successful fault detector model among the single-network models. A similar outcome was attained when the GRU was combined with other networks such as the CNN and the Attention, where the CNN-GRU and GRU-Attention models slightly outperformed the relevant models combined with the LSTM. 

However, the observed differences in the performance of these models are not statistically significant to draw definite conclusions. The LSTM and GRU have very similar structures and function in similar ways, making it difficult to pinpoint the reasons for the GRU's superiority in these tasks. Therefore, this could be due to a more favorable hyperparameter combination, or the GRU’s gated operations better fitting the characteristics of the used datasets. 

When it comes to the Attention-based models though, the GRU and LSTM acted as the encoders in the relevant architectures. Therefore, the slightly better performance of the GRU-Attention network could be attributed to the GRU’s memory structure, which exposes its full memory at each timestep, providing more information for the Attention mechanism to consider when assigning the importance weights. In contrast, the LSTM controls the level of its memory exposure through the Output gate, potentially limiting the information available to the Attention mechanism. However, this control can sometimes be beneficial depending on the task and specific sequence characteristics. 

The CNN model exhibited comparable performance to that of the advanced recurrent networks. Its efficiency can be attributed to the use of filters, which can capture local patterns such as spikes and trends that may be indicative of specific fault conditions in the rotating machine and the industrial gearbox. Moreover, by classifying their states based on extracted features from the input sequences, the impact of noisy sections on the classification decision can be reduced. 

While the results of the first classification task are consistent with some previous studies in the domain [3, 43], showing LSTM outperforming CNN, the second task contradicts these findings, as the CNN outperformed the LSTM in detecting the gearbox states. Although the performance differences of the CNN and LSTM in both tasks are not statistically significant to draw solid conclusions, this may be due to the lengthier sequences in the MaFaulDa dataset. Specifically, time dependencies between observations can be longer in range in the MaFaulDa dataset, and 

103 

therefore the LSTM can capture these dependencies more effectively than the CNN, which primarily focuses on extracting local features based on observations within the kernel size. 

Although models that rely on a single-network architecture demonstrated satisfying fault detection capabilities in both tasks, their performance was still inferior to those consisting of more than one DL technology. Characteristically, when the Attention mechanism was applied to the hidden states of the LSTM and GRU, the rates increased compared to when these models operated in isolation. Similarly, incorporating CNN and advanced recurrent networks into a hybrid architecture resulted in increased performance compared to using GRU, LSTM, and CNN as standalone models. This aligns with some previous research in the domain [18, 40, 47, 66], where hybrid networks exhibited higher fault detection rates compared to single-network models in other types of industrial equipment. 

The enhanced performance of models combining multiple DL technologies can be attributed to their deeper structure, enabling them to learn multiple levels of representation, with each layer building on the previous one. This allows fault patterns in the rotating machine and industrial gearbox to be captured more comprehensively. Additionally, these models leverage the strengths of different DL technologies. For instance, in CNN-LSTM and CNN-GRU models, the CNN identifies local patterns in sequence segments, while the LSTM and GRU capture time dependencies between the extracted features. Similarly, the Attention mechanism assigns varying weights to different sequence segments based on their informativeness by accessing the hidden states of LSTM and GRU. This approach enhances the ability to make accurate fault detection decisions, even in the presence of noise and redundant information. 

Another notable observation is that in both classification tasks, the Attention-based models slightly outperformed their counterparts combined with CNN. While both CNN and Attention aim to identify the most informative parts of a sequence to minimize the impact of redundant information, they employ different methods to achieve this. 

As part of their internal process, the CNN-LSTM and CNN-GRU models partition each input sequence into equal, fixed-length segments. Relevant features are selected from each segment and then provided to the corresponding recurrent network to capture time dependencies between the extracted features. In this process, the CNN applies fixed-length convolutions uniformly across every segment of the input sequence, regardless of the informativeness of its content. The downside of this fixed-length approach is that the generated feature maps may still include features extracted from relatively less informative or noisy segments. 

On the contrary, when the Attention mechanism dynamically assigns importance weights to each segment, it can consider the entire input sequence. This allows less informative segments, in terms of detecting the target fault state, to be entirely ignored, potentially leading to more efficient feature selection from the input sequences. Additionally, this gives the benefit to the Attention-based models to learn fault patterns by having a more comprehensive view of trends and time dependencies across the entire sequence, rather than relying solely on local observations within each fixed-length segment. 

In line with previous research in the field [48, 49, 50, 67], the Transformer in this study emerged as the best fault detector in both classification tasks, outperforming all other DL networks. The model was able to detect the largest number of fault conditions in the rotating machine and the industrial gearbox, which can be attributed to its deep and sophisticated structure and the use of a multi-head Attention mechanism. 

More specifically, the heads attend independently to different segments of each input sequence from the MaFaulDa and CSU datasets. This allows the Transformer to capture local 

104 

dependencies and relationships between observations in each segment. At the same time, the mechanism considers the outputs of all heads in parallel to learn global dependencies, ensuring that the model has a holistic understanding of the entire sequence and its context. This way, both local and global patterns are taken into consideration when identifying the fault states of the rotating machine and the industrial gearbox. 

Another important advantage of the Transformer is that it utilizes a normalization layer in each multi-head Attention block to normalize the inputs from the previous blocks. This makes the model more robust to outliers and noise in the input sequences. Additionally, the normalization layers can reduce its sensitivity to variations in inputs from previous layers, leading to better generalization, as evidenced by the successful classification of unseen sequences in both tasks. 

Similarly, by utilizing residual connections, the Transformer is able to bypass certain sub-layers that may not be contributing useful information for detecting the target class, making it easier to focus on the most informative parts of the sequence. Residual connections can also facilitate information flow, along with the propagation of gradients during training and prevent them from being vanished, allowing the Transformer to maintain an overview of the long-term dependencies in the sequences of the MaFaulDa and CSU datasets. 

While training times do not directly affect the effectiveness of fault detection models, the ability to rapidly train and update these models on the latest data is a distinct advantage. This allows for quicker adaptation to changing conditions in industrial environments, ensuring that the models remain effective and relevant, therefore this topic is discussed in this section. Among the models evaluated, significant variations were observed when applied to time series data from industrial environments. 

Advanced recurrent networks like LSTM and GRU, whether standalone or integrated into architectures like Attention-LSTM, and Attention-GRU, generally exhibited longer training times due to their sequential data processing, where each new input depends on the previous timestep. Among the recurrent networks, RNN was the fastest to train, with GRU requiring less time than LSTM due to its simpler structure and fewer parameters. In contrast, CNN-based models benefited from parallel filter operations in the convolutional and pooling layers, reducing the input representation size early in the process. This allowed CNN-based models to achieve shorter training times, and the standalone CNN to be the quickest to train in both tasks . 

Attention-based models had the longest training times, largely because recomputing Attention weights for all past hidden states in LSTM and GRU adds computational overhead. On the other hand, the Transformer model, despite its complexity, demonstrated rapid convergence, achieving high performance with fewer iterations. Its efficiency is due to parallel computations and the use of residual connections, which kept training times competitive with other models . 

Finally, in contrast to the training times, the inference times of the models differed only slightly (up to 2.5 seconds), suggesting comparable real-time or near-real-time fault detection capabilities. However, the timely aspect of fault predictions is not the focus of this study. Given the variability in industrial settings, "timely response" expectations vary with the process's intensity, safety needs, and the type of industrial equipment. Moreover, advancements in computational power beyond what was used in this study could enhance these models for more rigorous real-time applications. Nevertheless, deploying these models requires in-depth evaluation with industry experts to validate their effectiveness and reliability in meeting the specific operational requirements of the intended industrial machinery. 

105 

## **7. Conclusion and Future Work** 

## **7.1 Conclusion** 

In this study, the problem of fault detection in industrial machines was addressed by evaluating the suitability of multiple DL solutions for detecting fault states in a rotating machine and an industrial gearbox. The models were trained and tested on datasets consisting of multivariate time series. The first dataset included samples from five fault states and the normal operating state of the rotating machine, while the second dataset comprised samples from four gear fault states and the normal operating condition of the industrial gearbox. 

The experimental results demonstrated that the DL models were effective in detecting faults in both the rotating machine and the industrial gearbox, with eight out of the nine explored solutions achieving accuracy scores higher than 97%. Their success in the conducted experiments shows that DL models can handle the variability and complexity inherent in different types of industrial machines, suggesting their potential for application in the domain across a wide range of industrial machinery. This adaptability stems from the models' ability to learn and generalize from complex multivariate time series data, which is typical across many types of industrial equipment. 

Additionally, given the complexity of faults in industrial equipment and the substantial risks associated with unplanned downtime, such as operational disruptions and safety hazards, the results of this study are significant. The ability of DL models to accurately detect and classify a variety of fault types can enhance the reliability of industrial machinery and help prevent costly and potentially dangerous failures. By enabling earlier and more precise fault detection, these models can contribute to more effective maintenance strategies, potentially reducing the likelihood of unplanned downtime and minimizing safety risks. 

Furthermore, integrating these DL models into existing machinery maintenance frameworks can support the transition from reactive to predictive maintenance. This shift may lead to longer equipment lifespans, reduced maintenance costs, and a safer working environment. The results of the models also show great potential for eliminating increased costs arising from post-failure machinery maintenance. By adopting a more proactive approach, industries have the opportunity to optimize their operations, ensure continuous production, and maintain higher safety standards, ultimately enhancing overall operational efficiency. 

The high performance achieved in both classification tasks revealed that the DL models were not limited to merely distinguishing between anomalous and normal conditions; they could also identify the specific types of faults present in the equipment with high accuracy. This indicates their capability to learn detailed patterns of various fault types and their flexibility to adapt to new challenges arising from changes in industrial environments. 

This capability benefits traditional condition monitoring methods by merging the detection and diagnosis phases into a single, streamlined process. This integration can reduce the need for separate diagnostic tools and complex workflows, leading to a potential reduction in both complexity and computational costs. As a result, industries may achieve more efficient and cost-effective monitoring systems that not only detect faults more reliably but also diagnose them quickly, thereby minimizing downtime and enhancing productivity. 

Additionally, the models were successful in distinguishing between normal and fault conditions, which reduces the number of false alarms and avoids unnecessary costs and operational disruptions. This capability is critical in industrial settings where false alarms can 

106 

lead to unnecessary machine shutdowns, unplanned maintenance activities, and the diversion of resources to address issues that do not actually exist. By accurately differentiating between normal operations and genuine faults, these models can help to maintain operational continuity and ensure that maintenance efforts are focused only where they are truly needed. 

Reducing false alarms not only minimizes the risk of disrupting production but also enhances the efficiency of maintenance teams by allowing them to concentrate on actual issues, thereby improving the overall reliability and safety of operations. In addition, by preventing unnecessary interventions, these models can contribute to more efficient machinery operation and extend the useful life of equipment by avoiding premature maintenance actions. This ability also supports better inventory management by ensuring that spare parts and other resources are utilized efficiently, thus reducing waste and contributing to better sustainability practices. 

The comparative analysis conducted with previous studies that employed traditional ML highlighted that DL networks can achieve high performance while utilizing simpler and less elaborate data preprocessing techniques. This efficiency in data processing is particularly advantageous for real-time industrial applications, where the ability to quickly and accurately detect faults is crucial for ensuring continuous operations. In environments where machinery is required to function continuously, the reduction in preprocessing complexity not only has the potential to speed up the overall fault detection process but also minimize the latency between data collection and decision-making. 

This reduced latency can contribute to faster identification and resolution of issues, which is critical in preventing minor problems from escalating into major failures. Additionally, the simplified preprocessing may enable easier integration of DL models into existing systems by reducing the computational load associated with data preprocessing, making real-time or nearly real-time deployment more feasible. This is especially beneficial in the industrial sector, where even minor delays in detecting faults can lead to significant production losses, safety risks, and increased operational costs. 

On the other hand, the explored DL models were not entirely free from the need of data preprocessing. Normalization was essential for achieving the high accuracy observed in this study. As experienced in the process of the experimental phase, when raw data was provided directly to the models without normalization, the accuracy was lower, not drastically, but still noticeably so. This underscores the importance of even basic preprocessing techniques in enhancing the performance of DL models, indicating that some level of preprocessing may be necessary to achieve high accuracy. Especially in real-world industrial environments where data may be more variable, with higher levels of noise or missing values, additional and more complex preprocessing might be needed to maintain high accuracy and reliability. 

Evaluating the suitability of the explored DL models on the domain, showed that while all captured fault patterns effectively, some outperformed others in the conducted faut detection tasks. NLP-based technologies like the Attention mechanism and Transformer excelled, as did CNNs from computer vision and recurrent models like LSTM and GRU. Hybrid models further enhanced fault detection by combining the benefits of more than one DL technologies. However, small differences in accuracy make it challenging to draw definitive conclusions about their comparative effectiveness. 

Their relative performance may vary depending on the type of industrial machinery, the specific environment, and the nature of the data generated by the machines. For instance, CNN-based models might excel in scenarios where local pattern recognition is crucial, such as detecting specific fault signatures within shorter time frames or more localized sensor readings. This makes them particularly effective in environments requiring quick, localized pattern detection, 

107 

such as high-speed manufacturing lines. Moreover, their ability to reduce data dimensionality early in the processing pipeline allows for faster retraining cycles, making them well-suited for environments with swiftly evolving conditions, and process monitoring settings where it is imperative to keep the real-time fault detector updated with the latest training data due to rapidly shifting operational settings and parameters. 

Attention-based models and Transformers, which excel in processing and interpreting longrange dependencies within sequences, might be more beneficial in environments where fault patterns are spread across longer sequences or where particularly high accuracy rates are needed. These models are especially valuable in industries such as aerospace, nuclear power, or critical infrastructure, where precision and reliability are paramount. Recurrent models may be suitable for environments where time-dependent sequences play a significant role in fault evolution, and it is key to detecting and predicting failures. This variability highlights the importance of selecting the appropriate fault detection model based on the characteristics of the data, the machinery, and the operational environment, to ensure effectiveness. 

However, the process of implementing these models is not without its challenges. Based on some observations during the experimental phase of this study, similar issues might also emerge in the integration of these models into real-world industrial settings. One significant challenge is the lengthy training times required by DL networks. While these models are intended to be trained offline, which mitigates the impact to some extent, they can still be limiting for machines with rapidly changing operating conditions, where frequent retraining on the latest data is imperative. In such environments, the models need to be frequently updated with the latest data, which can be a time-consuming process. 

Moreover, these models are resource-intensive, potentially increasing costs for manufacturers due to the need for additional hardware, computational resources, and infrastructure to support the demanding training processes. Fine-tuning these models and adjusting hyperparameters to achieve high accuracy can be labor-intensive and requires specialized expertise, which is particularly crucial in cases where the parameters and conditions of industrial machines are constantly changing, necessitating ongoing model adjustments. 

Finally, another challenge observed was related to the explainability and interpretability of the models' decisions. In industrial settings, where operators and engineers need to understand and trust the decisions made by fault detection systems, the opaque nature of DL models can be a barrier. Explaining the reasons behind specific decisions requires detailed troubleshooting and expertise, which may not always be readily available. This complexity underscores the importance of not only training and fine-tuning models effectively, but also developing tools and methods for better interpretability of the model outputs. 

## **7.2 Contribution** 

Overall, this study contributes to advancing the field by expanding the knowledge base and providing practical guidelines for implementing DL techniques in industrial fault detection scenarios. By evaluating a diverse range of modern DL technologies across two different types of machinery, this research not only assesses the behavior and consistency of these models but also provides a broad comparison that is rare in existing literature. By advancing our knowledge of how various DL technologies can be effectively applied, this evaluation offers valuable insights that assist practitioners in selecting suitable DL approaches for real-world industrial settings. 

108 

Furthermore, the performance of DL models was compared with traditional ML models from previous research using the same datasets. Unlike some earlier studies that focused solely on performance evaluation, this research also emphasizes the importance of preprocessing techniques. Understanding the impact of these techniques can help practitioners make informed decisions, enhancing the practical implementation of fault detection solutions, an area that has been less addressed in prior literature. 

Finally, the transparency maintained throughout the experimental phase, detailing the design, implementation, preprocessing, and parameterization of models, is another key contribution. This level of detail, rarely found in existing literature, enhances the reproducibility and reliability of the findings, making the results more applicable and sustainable for real-world scenarios and future research. This approach enriches the domain's knowledge by bridging the gap between theoretical models and their practical applications. 

## **7.3 Limitations and Future Work** 

For the experimental part of this work, the explored DL networks were trained in a supervised manner, requiring labeled datasets. Since the faults in these datasets were predefined, this study did not investigate the effectiveness of the examined solutions in capturing novel fault types not previously encountered during the learning process. It is possible that an industrial machine might experience a new type of fault that was not anticipated during the data collection and labeling phase. 

In dynamic industrial environments where operating conditions constantly change, machines may encounter new anomalies and faults. To address these challenges, future research could explore unsupervised learning techniques, such as autoencoders and hierarchical clustering, which could be applied to classification tasks involving unknown fault types. 

Additionally, both datasets used in the classification tasks to evaluate the suitability of the explored DL models on the domain, were generated in test rigs under controlled conditions. While this study assumes that the simulations reflect the actual operating conditions of the relevant machines in real-world industrial environments, the high quality of the simulated data likely contributed to the strong performance of the models. 

However, in real-world industrial environments where conditions can be more challenging, the models' performance might differ. These challenges often introduce complexities that could be reflected in the relevant collected datasets. To further assess the applicability of the examined DL models in the domain, future research could incorporate datasets obtained directly from real-world industrial processes. 

Furthermore, a rotating machine and an industrial gearbox are chosen as examples of common industrial equipment to evaluate the suitability of DL solutions in the domain. Since the nature of industrial machines varies systematically along with the produced fault types, it is not guaranteed that the results from these specific solutions can be generalized to all types of industrial equipment. Therefore, future research could verify the promising results of the models on a wider range of industrial machinery. 

The comparison between the explored DL solutions and ML models was based solely on reported accuracy scores and preprocessing techniques. The lack of comprehensive evaluation beyond accuracy in most publications from tables 6.1 and 6.2 limited the depth of this comparison. Additionally, a constrained number of ML techniques were analyzed due to limited 

109 

literature access. To draw more concrete conclusions, future research should conduct a more exhaustive analysis by comparing a broader range of DL and ML models, enhancing our understanding of how DL solutions compare to traditional ML methods in this domain. 

Additionally, class imbalance remains a significant challenge in fault detection, as some fault types may be rare or costly to produce, resulting in limited training samples. To address this issue, future research could explore using Generative Adversarial Networks (GANs) for data augmentation to examine their impact on enhancing models’ performance. 

Finaly, the deployment of DL models in safety-critical systems presents notable challenges due to their "black-box" nature, which can obscure the decision-making processes. This limitation is particularly significant in industrial settings where the reliability and interpretability of fault detection systems are paramount. Future research could focus on enhancing the transparency of DL models through the integration of explainable AI techniques. 

110 

## **8. References** 

**[1]** Carvalho, T.P.; Soares, F.A.; Vita, R.; Francisco, R.D.; Basto, J.P.; Alcalá, S.G., “A systematic literature review of machine learning methods applied to predictive maintenance.” Comput. Ind. Eng. 2019, 137, 106024. 

**[2]** R. Naren and J. Subhashin, “Comparison of deep learning models for predictive maintenance”, 2020, IOP Conf. Ser.: Mater. Sci. Eng. 912 022029. 

**[3]** Femke Jansen, Mike Holenderski, Tanir Ozcelebi, “Predicting machine failures from industrial time series data”, 2018, Department of Mathematics and Computer Science Technische Universiteit Eindhoven, Den Dolech 2, 5600 AZ Eindhoven, The Netherlands. 

**[4]** Klaus-Dieter Thoben, Stefan Alexander WiesnerStefan, WuestThorsten Wuest, "Industrie 4.0 and Smart Manufacturing – A Review of Research Issues and Application Examples", 2017, International Journal of Automation Technology 11(1):4-19, DOI: 10.20965/ijat.2017.p0004. 

**[5]** Alkan, B., Vera, D.A., Ahmad, M., Ahmad, B. and Harrison, R., “Complexity in manufacturing systems and its measures: a literature review”, 2018, European J. Industrial Engineering, Vol. 12, No. 1, pp.116–150. 

**[6]** Shuai Zheng, Ahmed Farahat, Chetan Gupt, “Generative Adversarial Networks for Failure Prediction”, 2019, Industrial AI Lab, Hitachi America Ltd Santa Clara, CA, USA. 

**[7]** Park, You-Jin, Shu-Kai S. Fan, and Chia-Yu Hsu., "A Review on Fault Detection and Process Diagnostics in Industrial Processes", 2020, Processes 8, no. 9: 1123. https://doi.org/10.3390/pr8091123. 

**[8]** Gilchrist, “A. Industry 4.0: The Industrial Internet of Things; Springer: Berlin/Heidelberg, Germany”, 2016, ISBN 978-1-4842-2047-4. 

**[9]** Kumar, Arun & Kumar, Satyam, “Industry 4.0: Evolution, Opportunities and Challenges.”, 2020, DOI: 2455-2992. 

**[10]** Yongyi Ran, Xin Zhou, Pengfeng Lin, Yonggang Wen, Ruilong Deng, “A Survey of Predictive Maintenance: Systems, Purposes and Approaches”, 2019, Cornell University, https://doi.org/10.48550/arXiv.1912.07383. 

**[11]** Angelopoulos, Angelos, Emmanouel T. Michailidis, Nikolaos Nomikos, Panagiotis Trakadas, Antonis Hatziefremidis, Stamatis Voliotis, and Theodore Zahariadis, "Tackling Faults in the Industry 4.0 Era—A Survey of Machine-Learning Solutions and Key Aspects" _Sensors_ 20, no. 1: 109, 2020. https://doi.org/10.3390/s20010109. 

**[12]** Mohd Saufi, Mohd Syahril Ramadhan & Ahmad, Zair Asrar & Leong, M. & Lim, Meng, “Challenges and Opportunities of Deep Learning Models for Machinery Fault Detection and Diagnosis: A Review.”, 2019, IEEE Access. PP. 1-1. 10.1109/ACCESS.2019.2938227. 

**[13]** Shruti Jadon, Jan Kanty, Ajit Patnakar, "Challenges and Approaches to Time series forecasting: A Survey", 2021, Ajit Patnakar's Lab. 

**[14]** José F. Torres, Dalil Hadjout, Abderrazak Sebaa, Francisco Martínez-Álvarez, and Alicia Troncoso,”Deep Learning for Time Series Forecasting: A Survey.Big Data.”, 2021 3-21. http://doi.org/10.1089/big.2020.0159. 

**[15]** Iqbal H. Sarker, "Machine Learning: Algorithms, Real-World Applications and Research Directions", 2021, SN Computer Science 2021, DOI: 10.1007/s42979-021-00592-x. 

111 

**[16]** Bryan Lim, Stefan Zohren, "Time Series Forecasting with Deep Learning: A Survey", 2021, Philosophical Transactions of the Royal Society A: Mathematical, Physical and Engineering Sciences, The Royal Society, https://doi.org/10.48550/arxiv.2004.13408. 

**[17]** Steven Elsworth, Stefan Güttel, "Time Series Forecasting Using LSTM Networks: A Symbolic Approach", 2020, Cornell University, https://doi.org/10.48550/arXiv.2003.05672 

**[18]** W. Zhang, T. Zhang, G. Cui and Y. Pan, "Intelligent Machine Fault Diagnosis Using Convolutional Neural Networks and Transfer Learning," in IEEE Access, vol. 10, pp. 5095950973, 2022, doi: 10.1109/ACCESS.2022.3173444. 

**[19]** Yao Li, "A Fault Prediction and Cause Identification Approach in Complex Industrial Processes Based on Deep Learning", 2021, Computational Intelligence and Neuroscience, vol. 2021, Article ID 6612342, 13 pages, 2021. https://doi.org/10.1155/2021/6612342. 

**[20]** Lahiru Wijayasingha, John A. Stankovic, "Robustness to noise for speech emotion classification using CNNs and attention mechanisms", 2012, Smart Health, Volume 19, 100165, ISSN 2352-6483, https://doi.org/10.1016/j.smhl.2020.100165. 

**[21]** Xuran Meng and Yuan Cao and Difan Zou, "Per-Example Gradient Regularization Improves Learning Signals from Noisy Data", 2023, 2303.17940, arXiv, stat.ML https://doi.org/10.48550/arXiv.2303.17940. 

**[22]** A Demidova, “Recurrent Neural Networks' Configurations in the Predictive Maintenance Problems”, 2020, IOP Conf. Ser.: Mater. Sci. Eng. 714 012005 

**[23]** Mateus, Balduíno C., Mateus Mendes, José T. Farinha, Rui Assis, and António M. Cardoso, "Comparing LSTM and GRU Models to Predict the Condition of a Pulp Paper Press", 2021, Energies 14, no. 21: 6958. https://doi.org/10.3390/en14216958. 

**[24]** https://blog.isa.org/worlds-largest-manufacturers-lose-1-trillion/year-to-machine-failure, last accessed: 15/08/2022. 

**[25]** Sanna Nenonen, “Fatal workplace accidents in outsourced operations in the manufacturing industry”, 2011, Safety Science, Volume 49, Issue 10, Pages 1394-1403, ISSN 0925-7535, https://doi.org/10.1016/j.ssci.2011.06.004. 

**[26]** C. Liu, D. Tang, H. Zhu and Q. Nie, "A Novel Predictive Maintenance Method Based on Deep Adversarial Learning in the Intelligent Manufacturing System.", 2021, in IEEE Access, vol. 9, pp. 49557-49575, 2021, DOI: 10.1109/ACCESS.2021.3069256. 

**[27]** https://medium.com/swlh/machine-learning-for-equipment-failure-prediction-andpredictive-maintenance-pm-e72b1ce42da1, last accessed: 15/08/2022. 

**[28]** Elshaboury N. Al-Sakkaf, A. Alfalah G. Abdelkader E.M, “Data-Driven Models for Forecasting Failure Modes in Oil and Gas Pipes.”, 2022, Processes 2022, 10, 400. https://doi.org/10.3390/pr10020400. 

**[29]** G Nath, Aneesh & Sharma, Anshul & Udmale, Sandeep & Singh, Sanjay, “An Early Classification Approach for Improving Structural Rotor Fault Diagnosis.”, 2021, IEEE Transactions on Instrumentation and Measurement. 70. 1-13. 10.1109/TIM.2020.3043959. 

**[30]** Sokolovsky, Artur, David Hare, and Jorn Mehnen. 2021. "Cost-Effective Vibration Analysis through Data-Backed Pipeline Optimisation" _Sensors_ 21, no. 19: 6678. https://doi.org/10.3390/s21196678. 

112 

**[31]** Martins, Dionisio Henrique Carvalho De Sá & Hemerly, Douglas De Oliveira & Marins, Matheus & Lima, Amaro & Silva, Fabricio & Prego, Thiago & Ribeiro, Felipe & Netto, Sergio & da Silva, Eduardo, “Application of Machine Learning to Evaluate Unbalance Severity in Rotating Machines: Vol. 2.”, 2019, 10.1007/978-3-319-99268-6_11. 

**[32]** A. A. de Lima et al., "On fault classification in rotating machines using fourier domain features and neural networks", 2013, IEEE 4th Latin American Symposium on Circuits and Systems (LASCAS), Cusco, Peru, 2013, pp. 1-4, doi: 10.1109/LASCAS.2013.6518984. 

**[33]** D. Pestana-Viana, R. Zambrano-López, A. A. de Lima, T. De M. Prego, S. L. Netto and E. A. B. da Silva, "The influence of feature vector on the classification of mechanical faults using neural networks", _2016, IEEE 7th Latin American Symposium on Circuits & Systems (LASCAS),_ Florianopolis, Brazil, 2016, pp. 115-118, doi: 10.1109/LASCAS.2016.7451023. 

**[34]** M. A. Ali, A. A. Bingamil, A. Jarndal and I. Alsyouf, "The Influence of Handling Imbalance Classes on the Classification of Mechanical Faults Using Neural Networks", _2019, 8th International Conference on Modeling Simulation and Applied Optimization (ICMSAO),_ Manama, Bahrain, 2019, pp. 1-5, doi: 10.1109/ICMSAO.2019.8880437. 

**[35]** Jarndal, Anwar & Alsyouf, Imad, “On the Usefulness of Pre-processing Methods in Rotating Machines Faults Classification using Artificial Neural Network”, 2020, Journal of Applied and Computational Mechanics. 7. 10.22055/JACM.2020.35354.2639. 

**[36]** R. Zhao, D. Wang, R. Yan, K. Mao, F. Shen and J. Wang, "Machine Health Monitoring Using Local Feature-Based Gated Recurrent Unit Networks", 2018, in _IEEE Transactions on Industrial Electronics,_ vol. 65, no. 2, pp. 1539-1548, doi: 10.1109/TIE.2017.2733438. 

**[37]** Mingxin Chen, Xinyu Pang, Kaibo LMingxin Chen, Xinyu Pang, Kaibo Lu, "Fault Diagnosis of Planetary Gearbox Based on Random Forest and Singular Value Difference Spectrum", 2019, Advances in Asset Management and Condition Monitoring - chapter 124, Springer International Publishing. 

**[38]** S. -N. Chen, F. Liu, C. -X. Gao and J. Li, "Gearbox Fault Diagnosis Classification with Empirical Mode Decomposition Based on Improved Long Short-Term Memory" _2021, IEEE 6th International Conference on Cloud Computing and Big Data Analytics (ICCCBDA),_ Chengdu, China, 2021, pp. 568-575, doi: 10.1109/ICCCBDA51879.2021.9442505. 

**[39]** Xiaojie Guo, Liang Chen, Changqing Shen, “Hierarchical adaptive deep convolution neural network and its application to bearing fault diagnosis.”, 2016, Measurement, Volume 93, Pages 490-502, ISSN 0263-2241, https://doi.org/10.1016/j.measurement.2016.07.054. 

**[40]** Ahmed Nasser, Huthaifa Al-Khazraji, “A hybrid of convolutional neural network and long short-term memory network approach to predictive maintenance.”, 2022, (IJECE)Vol.12, No.1, pp. 721~730ISSN: 2088-8708, DOI: 10.11591/ijece.v12i1.pp721-730, International Journal of Electrical and Computer Engineering. 

**[41]** Roberto M. Souza, Erick G.S. Nascimento, Ubatan A. Miranda, Wenisten J.D. Silva, Herman A. Lepikson, “Deep learning for diagnosis and classification of faults in industrial rotating machinery.” 2021, Computers and Industrial Engineering. 153:107060, ISSN 03608352. doi:10.1016/J.CIE.2020.107060. 

**[42]** K. S. Kiangala and Z. Wang, "An Effective Predictive Maintenance Framework for Conveyor Motors Using Dual Time-Series Imaging and Convolutional Neural Network in an Industry 4.0 Environment.", 2022, in IEEE Access, vol. 8, pp. 121033-121049, DOI: 10.1109/ACCESS.2020.3006788. 

113 

**[43]** Jinhao Lei, Chao Liu, Dongxiang Jiang, “Fault diagnosis of wind turbine based on Long Short-term memory networks”, 2019, Renewable Energy, Volume 133, Pages 422-432, ISSN 0960-1481, https://doi.org/10.1016/j.renene.2018.10.031. 

**[44]** Vaswani, Ashish & Shazeer, Noam & Parmar, Niki & Uszkoreit, Jakob & Jones, Llion & Gomez, Aidan & Kaiser, Lukasz & Polosukhin, Illia, “Attention Is All You Need”, 2017. 

**[45]** Li, Xiang & Zhang, Wei & Ding, Qian, “Understanding and Improving Deep LearningBased Rolling Bearing Fault Diagnosis with Attention Mechanism.”, 2019, Signal Processing. 161. 10.1016/j.sigpro.2019.03.019. 

**[46]** Xue, Xingsi Xie, Yuan Zhao, Jisheng Qiang, Baohua Mi, Luzhong Tang, Chenghua Li, Longge, “Attention Mechanism-Based CNN-LSTM Model for Wind Turbine Fault Prediction Using SSN Ontology Annotation.”, 2021, Wireless Communications and Mobile Computing Hindawi, https://doi.org/10.1155/2021/6627588. 

**[47]** Jin, Yuhong and Hou, Lei and Chen, Yushu, “A new rotating machinery fault diagnosis method based on the Time Series Transformer.”, 2021, arXiv, DOI:10.48550/ARXIV.2108.12562. 

**[48]** Nascimento, Erick Giovani Sperandio and Liang, Julian Santana and Figueiredo, Ilan Sousa and Guarieiro, Lilian Lefol Nani, “T4PdM: a Deep Neural Network based on the Transformer Architecture for Fault Diagnosis of Rotating Machinery.”, 2022, arXiv, DOI:10.48550/ARXIV.2204.03725. 

**[49]** Hui Wang, Jiawen Xu, Ruqiang Yan, Chuang Sun, Xuefeng Chen, “Intelligent Bearing Fault Diagnosis Using Multi-Head Attention-Based CNN.”, 2020, Procedia Manufacturing, Volume 49, Pages 112-118, ISSN 2351-9789, https://doi.org/10.1016/j.promfg.2020.07.005. 

**[50]** Majdi Mansouri et al, "Process Monitoring Using Data-Based Fault Detection Techniques: Comparative Studies", 2016, Fault Diagnosis and DetectionChapter: Fault Diagnosis and DetectionPublisher: InTechEditors: Martina Usljebrka. 

**[51]** Raparthi Srilakshmi, Ch.Ratnam, Vana Vital, "A Review on Fault Detection, Diagnosis and Prognosis, in Vibration Measurement through Wavelets on Machine Elements", 2019, International Journal of Applied Engineering Research ISSN 0973-4562 Volume 14, Number 2 (2019) pp. 547-555. 

**[52]** Xu, K., Li, Y., Liu, C. et al. “Advanced Data Collection and Analysis in Data-Driven Manufacturing Process”, 2020, Chin. J. Mech. Eng. 33, 43 (2020), https://doi.org/10.1186/s10033-020-00459-x 

**[53]** Khaoula Tidriri, Nizar Chatti, Sylvain Verron, Teodor Tiplica, "Bridging data-driven and model-based approaches for process fault diagnosis and health monitoring: A review of researches and future challenges", 2016, Formerly Annual Review in Automatic Programming A Journal of IFAC, the International Federation of Automatic Control, ISSN: 1367-5788. 

**[54]** Kristen Severson, Paphonwit Chaiwatanodom, Richard D. Braatz, "Perspectives on process monitoring of industrial systems", 2016, Massachusetts Institute of Technology, 77 Massachusetts Avenue, Cambridge, MA 02139, USA, IFAC, the International Federation of Automatic Control. 

**[55]** Park, Donghyun, Seulgi Kim, Yelin An, and Jae-Yoon Jung. 2018. "LiReD: A LightWeight Real-Time Fault Detection System for Edge Computing Using LSTM Recurrent Neural Networks" Sensors 18, no. 7: 2110. https://doi.org/10.3390/s18072110. 

114 

**[56]** Aleksandar Brkovic, Dragoljub Gajic, Jovan Gligorijevic, Ivana Savic-Gajic, Olga Georgieva, Stefano Di Gennaro, “Early fault detection and diagnosis in bearings for more efficient operation of rotating machinery”, 2017, Energy, Volume 136, Pages 63-71, ISSN 0360-5442, https://doi.org/10.1016/j.energy.2016.08.039. 

**[57]** Baig, Shuja-ur-Rehman & Iqbal, Waheed & Berral, Josep & Erradi, Abdelkarim & Carrera, David, “Adaptive Prediction Models for Data Center Resources Utilization Estimation.”, 2019, IEEE Transactions on Network and Service Management. 16. 10.1109/TNSM.2019.2932840. 

**[58]** Sarker, Iqbal H. “Deep Learning: A Comprehensive Overview on Techniques, Taxonomy, Applications and Research Directions.”, 2021, SN Computer Science, Springer. 

**[59]** Daniel Godoy, "Understanding binary cross-entropy / log loss: a visual explanation", 2018, Published in Towards Data Science, https://towardsdatascience.com/understanding-binarycross-entropy-log-loss-a-visual-explanation-a3ac6025181a, last accessed: 14/08/22. 

**[60]** Ian Goodfellow, Yoshua Bengio, and Aaron Courville, “Deep Learning”, 2016, MIT Press. http://www.deeplearningbook.org, last accessed: 15/08/22. 

**[61]** Bahdanau, Dzmitry and Cho, Kyunghyun and Bengio, Yoshua, "Neural Machine Translation by Jointly Learning to Align and Translate.", 2014, arXiv, https://doi.org/10.48550/arXiv.1409.0473. 

**[62]** Marco Del Pra, Time Series Forecasting with Deep Learning and Attention Mechanism, 2020, Published in Towards Data Science, https://towardsdatascience.com/time-seriesforecasting-with-deep-learning-and-attention-mechanism-2d001fc871fc, last accessed: 16/08/2022. 

**[63]** http://www02.smt.ufrj.br/~offshore/mfs/page_01.html, last accessed: 19/08/2022. 

**[64]** https://github.com/cathysiyu/Mechanical-datasets, last accessed: 12/04/2023 

**[65]** S. Cao, K. Zhang, Y. An and L. Wang, "An intelligent fault diagnosis method for planetary gearbox based on the MPSO optimizing the Bi-LSTM parameter combination," 2021 CAA Symposium on Fault Detection, Supervision, and Safety for Technical Processes (SAFEPROCESS), Chengdu, China, 2021, pp. 1-6, doi: 10.1109/SAFEPROCESS52771.2021 .9693702. 

**[66]** Q. Zhou and J. Tang, "An Interpretable Parallel Spatial CNN-LSTM Architecture for Fault Diagnosis in Rotating Machinery,", July 2024, in IEEE Internet of Things Journal _,_ doi: 10.1109/JIOT.2024.3422969. 

**[67]** W. Sun, R. Yan, R. Jin, J. Xu, Y. Yang and Z. Chen, "LiteFormer: A Lightweight and Efficient Transformer for Rotating Machine Fault Diagnosis," in IEEE Transactions on Reliability, vol. 73, no. 2, pp. 1258-1269, June 2024, doi: 10.1109/TR.2023.3322860. 

## **Figure Citation** 

**Note:** _This section includes references for images used that are not contained in scientific papers previously cited in this document. These references are provided to acknowledge the sources of original images and illustrations developed for or adapted specifically for this study._ 

**Figure 1.1:** Bahrin, Mohd, Mohd Fauzi Bin Othman, Nor Hayati Nor Azli and Muhamad Farihin Talib.“Industry 4.0: a review of industrial automation and robotic.”, 2016. 

115 

**Figure 1.6:** Wei Cui, Guoying Meng, Aiming Wang, Xinge Zhang, Jun Ding, "Application of Rotating Machinery Fault Diagnosis Based on Deep Learning", Shock and Vibration, vol. 2021, Article ID 3083190, 30 pages, 2021. https://doi.org/10.1155/2021/3083190. 

**Figure 3.6:** Kim, Ji-Won & Kim, Kwang-Jin & Lee, Soo-Jin & Kim, Yeong-Ho & Ahn, JiHye & Lee, Yangwon, “Prediction of Arctic Sea Ice Concentration of Kara-Barents Seas Using RCM Data with Machine Learning.”, 2017, Journal of Climate Research. 12. 349-365. 10.14383/cri.2017.12.4.3049. 

**Figure 3.7:** https://ripton-rosen.medium.com/gradient-descent-into-the-algorithm1b13b8b941cd, last accessed: 14/06/2022. 

**Figure 3.8:** https://towardsdatascience.com/whats-the-role-of-weights-and-bias-in-a-neuralnetwork-4cf7e9888a0f, last accessed: 17/06/2022. 

**Figures 3.12, 3.13, 3.14:** https://peltarion.com/knowledge-center/modeling-view/build-an-aimodel/blocks, last accessed: 16/06/2022. 

**Figure 3.19:** https://medium.com/@aatish_kayyath/confusion-matrix-lets-clear-thisconfusion-4b0bc5a5983c, last accessed: 18/06/2022. 

**Figure 4.4:** Hu, Chaofan & He, Shuilong & Wang, Yanxue. (2021). A classification method to detect faults in a rotating machinery based on kernelled support tensor machine and multilinear principal component analysis. Applied Intelligence. 51. 1-13. 10.1007/s10489-020-02011-9. 

**Cover figure:** https://smartmfg.me.wisc.edu/, last accessed: 26/03/2023. 

116 

