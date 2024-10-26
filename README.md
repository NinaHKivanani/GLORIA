<p align="center">
  <img src="https://github.com/NinaHKivanani/GLORIA/blob/main/img/logo_gloria.svg" alt="GLORIA Logo" width="200">
</p>

# GLORIA: GeneraLization with chain of thought fOR bIomedical Ai

## Objectives
<div align="justify">
This project aims to develop a benchmarking framework that evaluates the ability of large language models (LLMs) to generalize across key biomedical tasks, specifically Named Entity Recognition (NER) and diagnosis prediction. Using Chain of Thought (CoT) prompting, we will guide models through structured reasoning steps to improve both task generalization and interpretability. Enhanced interpretability is critical in healthcare, where trust and transparency are paramount.

### Key Goals:
1. **Cross-Task Transfer**: Determine if fine-tuning an LLM on NER can improve zero-shot or few-shot performance in diagnosis prediction.
2. **Interpretability**: Assess how CoT prompting improves interpretability, allowing clinicians to understand the model’s decision-making process.
3. **Model Evaluation**: Model Evaluation: Compare BioGPT and Llama3.2’s effectiveness in generalization across tasks with and without CoT prompting, providing insights into optimal architectures for biomedical applications. BioGPT, with its domain-specific training on biomedical literature, is expected to excel in tasks requiring clinical terminology and understanding, whereas Llama3.2’s flexibility in general-purpose reasoning may demonstrate strengths in cross-task adaptability.

Additionally, due to the computational constraints typical in a hackathon setting, we will use Parameter-Efficient Fine-Tuning (PEFT) and Low-Rank Adaptation (LoRa) techniques. These approaches allow us to fine-tune models with minimal computational cost, ensuring efficient experimentation and optimization within limited resources.

## Background and Motivation
Recent advancements in NLP have shown that LLMs can generalize across multiple tasks. Chain of Thought (CoT) prompting has demonstrated success in guiding LLMs through multi-step reasoning, particularly for complex problem-solving tasks. However, its application in biomedical AI remains underexplored, especially in task-specific domains like NER and diagnosis prediction, which require a nuanced understanding of clinical language and interpretability. The objective of this project is to fill this gap by evaluating CoT’s potential to enhance cross-task generalization and reasoning within biomedical contexts~\cite{wang2022formulating, weichain}.

While CoT has shown promise in other domains, this approach has not been thoroughly researched within the BLAH framework, and our project extends beyond previous work by specifically exploring its effectiveness in biomedical NER and diagnosis prediction tasks. This novel focus aims to bridge a critical gap in the use of LLMs for cross-task generalization in healthcare.

BioGPT, with its domain-specific training on biomedical literature, is expected to perform well on tasks that require deep knowledge of the clinical language and medical terminology~\cite{Luo2022BioGPT}. In contrast, Llama 3.2, a robust general-purpose LLM, brings flexibility and strong reasoning capabilities that make it suitable for testing cross-task generalization~\cite{nori2023capabilities}. By assessing how each model handles task transfer between NER and diagnosis prediction under CoT prompting, we aim to identify the ideal balance between domain-specific knowledge and broad generalization. This comparison will reveal architecture-specific strengths, highlighting whether domain specialization or broader adaptability yields better results in biomedical applications.

Given the hackathon’s time limitations, we will use PEFT and LoRa, which allow effective model adaptation without excessive computational demands. These techniques ensure that our approach remains feasible, allowing us to explore task transfer and interpretability comprehensively, even with limited infrastructure.

### Hypotheses
- **Cross-Task Transfer Hypothesis**: Cross-Task Transfer Hypothesis: Fine-tuning on NER will improve zero-shot performance in diagnosis prediction, reducing the need for costly retraining.
- **Interpretability Hypothesis**: CoT prompts will enhance the interpretability of model outputs, making the model’s reasoning more accessible and understandable for healthcare professionals.
  
## Research Questions
1. Can fine-tuning NER improve model performance on diagnosis prediction without additional training?
2. How does CoT prompting enhance interpretability in biomedical contexts, specifically in reasoning tasks?
3. How do recent LLM architectures (BioGPT, Llama3.2) compare in generalization across biomedical tasks when CoT is applied?

## Approach and Tasks
This project will conduct two core experiments to evaluate cross-task generalization with and without CoT prompting:

1. **NER to Diagnosis Prediction**: Fine-tune a model on NER and evaluate its performance in diagnosis prediction under zero-shot and few-shot settings.
2. **Diagnosis Prediction to NER**: Conversely, fine-tune the model on diagnosis prediction and assess generalization to NER, focusing on how CoT improves interpretability.

To guide reasoning, CoT prompts will be designed to structure biomedical tasks logically. For instance, NER prompts will decompose entity identification into steps, while diagnosis prompts will guide symptom assessment to diagnosis.

## Datasets
To ensure robust and relevant evaluation, we will use well-established biomedical datasets:

- **Diagnosis Prediction**:
   - **CheXpert Dataset**: Chest X-rays annotated with diagnoses, useful for evaluating diagnosis prediction in radiology.
   - **MIMIC-CXR**: Contains radiology reports with associated diagnostic labels, ideal for both image-based and text-based evaluations.
- **NER**:
   - **BC5CDR**: PubMed abstracts annotated for chemical-disease relationships, suited for biomedical NER.
   - **NCBI Disease Corpus**: PubMed abstracts with disease-specific annotations.

## Evaluation Metrics
To measure model performance and interpretability, we will employ standardized metrics for each task:
- **NER (F1-Score)**: Balances precision and recall to assess entity extraction accuracy. This metric is especially valuable due to the imbalance and specificity required in biomedical datasets, where certain entity types, such as rare diseases or chemical names, are underrepresented~\cite{lee2020biobert}.
- **Diagnosis Prediction (AUC-ROC: Area Under the Curve - Receiver Operating Characteristic)}~\cite{irvinchexpert}**: It evaluates a model’s capacity to differentiate among diagnostic categories, capturing how well it distinguishes between true positive and false positive rates across various thresholds. In clinical contexts, the AUC-ROC provides an indication of the model's reliability in predicting outcomes across multiple conditions, such as distinguishing between “normal” and “pneumonia” or “cardiomegaly” categories in radiological imaging.

## 5-Day Hackathon Plan
<details>
<summary>Click to view details</summary>

### Day 1: Initialization and Setup
   - Define project goals, assign roles, and discuss tasks (NER and diagnosis prediction).
   - Set up the environment and prepare datasets.

### Day 2: CoT Prompt Engineering and Baseline Training
   - Design CoT prompts and train baseline LLMs (Llama3.2, BioGPT) on NER without CoT prompting.

### Day 3: Enhanced CoT Prompting and Cross-Task Evaluation
   - Implement dynamic CoT prompting and test cross-task generalization.

### Day 4: Few-Shot Learning and Tuning
   - Introduce few-shot learning across tasks and fine-tune models using PEFT and LoRa approaches.

### Day 5: Final Testing, Analysis, and Presentation
   - Perform final evaluations, aggregate metrics, and prepare visualizations.
</details>

## Conclusion
This project leverages Chain of Thought prompting to test cross-task generalization and interpretability in biomedical AI. By assessing CoT’s role in reasoning, we provide insights into LLM capabilities in real-world healthcare applications.

## Expected Contributions
1. **Benchmarking Framework**: A tool for researchers to evaluate LLM generalization in biomedical tasks.
2. **Improved Interpretability**: Insights into how CoT prompting aids reasoning in high-stakes clinical applications.

## Team

<table>
  <tr>
    <td><img src="img/nnina.png" width="95" /></td>
    <td><strong>Nina Hosseini-Kivanani</strong><br>University of Luxembourg, Faculty of Science, Technology and Medicine (FSTM), Department of Computer Science</td>
  </tr>
  <tr>
    <td><img src="img/dimitra.png" width="95" /></td>
    <td><strong>Dimitra Anastasiou</strong><br>Luxembourg Institute of Science and Technology (LIST)</td>
  </tr>
  <tr>
    <td><img src="img/Davide-Liga.jpg" width="95" /></td>
    <td><strong>Davide Liga</strong><br>University of Luxembourg, Faculty of Science, Technology and Medicine (FSTM), Department of Computer Science </td>
  </tr>
</table>

## References

<details>
<summary>Click to view references</summary>

1. **Irvin, J., et al.** (2019). CheXpert: A large chest radiograph dataset with uncertainty labels and expert comparison. *Proceedings of the AAAI Conference on Artificial Intelligence*, 33, 590–597. [Link](https://arxiv.org/abs/1901.07031)

2. **Dogan, R. I., Leaman, R., and Lu, Z.** (2014). NCBI disease corpus: A resource for disease name recognition and concept normalization. *Journal of Biomedical Informatics*, 47, 1–10. [Link](https://doi.org/10.1016/j.jbi.2013.12.006)

3. **Lee, J., Yoon, W., Kim, S., Kim, D., Kim, S., So, C. H., and Kang, J.** (2020). BioBERT: A pre-trained biomedical language representation model for biomedical text mining. *Bioinformatics*, 36(4), 1234–1240. [Link](https://doi.org/10.1093/bioinformatics/btz682)

4. **Li, J., Sun, Y., Johnson, R. J., Sciaky, D., Wei, C.-H., Leaman, R., Davis, A. P., Mattingly, C. J., Wiegers, T. C., and Lu, Z.** (2016). BioCreative V CDR task corpus: a resource for chemical disease relation extraction. *Database*, 2016. [Link](https://doi.org/10.1093/database/baw068)

5. **Nori, H., King, N., McKinney, S. M., Carignan, D., and Horvitz, E.** (2023). Capabilities of GPT-4 on medical challenge problems. [Link](https://arxiv.org/abs/2303.13375)

6. **Wang, Z., Zhao, K., Wang, Z., and Shang, J.** (2022). Formulating few-shot fine-tuning towards language model pre-training: A pilot study on named entity recognition. *Findings of the Association for Computational Linguistics: EMNLP 2022*, 3186–3199. [Link](https://doi.org/10.18653/v1/2022.findings-emnlp.233)

7. **Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi, E., Le, Q. V., Zhou, D., et al.** (2022). Chain-of-thought prompting elicits reasoning in large language models. *Advances in Neural Information Processing Systems*, 35, 24824–24837. [Link](https://arxiv.org/abs/2201.11903)
</details>

<hr>
`Note: The cover image/logo was created using OpenAI's DALL·E.`

</div>
