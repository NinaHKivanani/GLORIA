![Logo](https://github.com/NinaHKivanani/GLORIA/blob/main/img/logo_gloria.svg)

# GLORIA: GeneraLization with chain of thought fOR bIomedical Ai


## Objectives
This project aims to develop a benchmarking framework that evaluates the ability of large language models (LLMs) to generalize across key biomedical tasks, specifically Named Entity Recognition (NER) and diagnosis prediction. Using Chain of Thought (CoT) prompting, we will guide models through structured reasoning steps to improve both task generalization and interpretability. Enhanced interpretability is critical in healthcare, where trust and transparency are paramount.

### Key Goals:
1. **Cross-Task Transfer**: Determine if fine-tuning an LLM on NER can improve zero-shot or few-shot performance in diagnosis prediction.
2. **Interpretability**: Assess how CoT prompting improves interpretability, allowing clinicians to understand the model’s decision-making process.
3. **Model Evaluation**: Compare BioGPT and Llama3.2’s effectiveness in generalization across tasks with and without CoT prompting, providing insights into optimal architectures for biomedical applications. BioGPT, with its domain-specific training on biomedical literature, is expected to excel in tasks requiring clinical terminology and understanding, whereas Llama3.2’s flexibility in general-purpose reasoning may demonstrate strengths in cross-task adaptability.

Additionally, due to the computational constraints typical in a hackathon setting, we will use Parameter-Efficient Fine-Tuning (PEFT) and Low-Rank Adaptation (LoRa) techniques. These approaches allow us to fine-tune models with minimal computational cost, ensuring efficient experimentation and optimization within limited resources.

## Background and Motivation
Recent advancements in NLP have shown that LLMs can generalize across multiple tasks. Chain of Thought (CoT) prompting has demonstrated success in guiding LLMs through multi-step reasoning, particularly for complex problem-solving tasks. However, its application in biomedical AI remains underexplored, especially in task-specific domains like NER and diagnosis prediction, which require a nuanced understanding of clinical language and interpretability.

Our project extends beyond previous work by specifically exploring CoT's effectiveness in biomedical NER and diagnosis prediction tasks, filling a critical gap in healthcare LLM research.

* **Hypotheses**:
   1. **Cross-Task Transfer Hypothesis**: Fine-tuning on NER will improve zero-shot performance in diagnosis prediction, reducing the need for costly retraining.
   2. **Interpretability Hypothesis**: CoT prompts will enhance the interpretability of model outputs, making the model’s reasoning more accessible and understandable for healthcare professionals.

## Research Questions
1. Can fine-tuning NER improve model performance on diagnosis prediction without additional training?
2. How does CoT prompting enhance interpretability in biomedical contexts, specifically in reasoning tasks?
3. How do recent LLM architectures (BioGPT, Llama3.2) compare in generalization across biomedical tasks when CoT is applied?

## Approach and Tasks
This project will conduct two core experiments to evaluate cross-task generalization with and without CoT prompting:

1. **NER to Diagnosis Prediction**: Fine-tune a model on NER and evaluate its performance in diagnosis prediction under zero-shot and few-shot settings.
2. **Diagnosis Prediction to NER**: Conversely, fine-tune the model on diagnosis prediction and assess generalization to NER, focusing on how CoT improves interpretability.

## Datasets
To ensure robust and relevant evaluation, we will use well-established biomedical datasets:

- **Diagnosis Prediction**:
   - CheXpert Dataset: Chest X-rays annotated with diagnoses, useful for evaluating diagnosis prediction in radiology.
   - MIMIC-CXR: Contains radiology reports with associated diagnostic labels, which are ideal for both image-based and text-based evaluations.
- **NER**:
   - BC5CDR: PubMed abstracts annotated for chemical-disease relationships, suited for biomedical NER.
   - NCBI Disease Corpus: PubMed abstracts with disease-specific annotations.

## Evaluation Metrics
To measure model performance and interpretability, we will employ standardized metrics for each task:

- **NER (F1-Score)**: Balances precision and recall to assess entity extraction accuracy.
- **Diagnosis Prediction (AUC-ROC)**: Evaluates a model’s capacity to differentiate among diagnostic categories.

## 5-Day Hackathon Plan

- **Day 1: Initialization and Setup**
   - Define project goals, assign roles, and discuss tasks (NER and diagnosis prediction).
   - Set up the environment and prepare datasets.
   - Begin exploring simple CoT prompts for NER and diagnosis tasks.

- **Day 2: CoT Prompt Engineering and Baseline Training**
   - Design CoT prompts to guide reasoning in NER and diagnosis prediction.
   - Train baseline LLMs (Llama3.2, BioGPT) on NER without CoT prompting.
   - Initial zero-shot testing on diagnosis prediction.

- **Day 3: Enhanced CoT Prompting and Cross-Task Evaluation**
   - Implement dynamic CoT prompting for complex tasks.
   - Test cross-task generalization with CoT prompts applied to the trained model.

- **Day 4: Few-Shot Learning and Tuning**
   - Introduce few-shot learning across tasks.
   - Fine-tune models using PEFT and LoRa approaches, assessing CoT impact on transferability.

- **Day 5: Final Testing, Analysis, and Presentation**
   - Perform final evaluations and aggregate metrics (F1, AUC-ROC).
   - Prepare visualizations and present results, emphasizing CoT’s role in reasoning and task transfer.

## Conclusion
This project leverages Chain of Thought prompting to test cross-task generalization and interpretability in biomedical AI, focusing on NER and diagnosis prediction. By assessing CoT’s role in reasoning, we provide insights into LLM capabilities in real-world healthcare applications, offering a transparent approach that clinicians can trust.

## Expected Contributions
1. **Benchmarking Framework**: A tool for researchers to evaluate LLM generalization in biomedical tasks.
2. **Improved Interpretability**: Insights into how CoT prompting aids reasoning in high-stakes clinical applications.


## Team

<table>
  <tr>
    <td><img src="images/nina.jpg" width="100" /></td>
    <td><strong>Nina Hosseini-Kivanani</strong><br>University of Luxembourg, Faculty of Science, Technology and Medicine (FSTM), Department of Computer Science</td>
  </tr>
  <tr>
    <td><img src="images/dimitra.jpg" width="100" /></td>
    <td><strong>Dimitra Anastasiou</strong><br>Luxembourg Institute of Science and Technology (LIST)</td>
  </tr>
  <tr>
    <td><img src="images/davide.jpg" width="100" /></td>
    <td><strong>Davide Liga</strong><br>University of Luxembourg, Faculty of Science, Technology and Medicine (FSTM), Department of Computer Science (DCS)</td>
  </tr>
</table>




## References

1. **Irvin, J., Rajpurkar, P., Ko, M., Yu, Y., Ciurea-Ilcus, S., Chute, C., Marklund, H., Haghgoo, B., Ball, R., Shpanskaya, K., et al.** (2019). CheXpert: A large chest radiograph dataset with uncertainty labels and expert comparison. *Proceedings of the AAAI Conference on Artificial Intelligence*, 33, 590–597. [Link](https://arxiv.org/abs/1901.07031)

2. **Dogan, R. I., Leaman, R., and Lu, Z.** (2014). NCBI disease corpus: a resource for disease name recognition and concept normalization. *Journal of Biomedical Informatics*, 47, 1–10. [Link](https://doi.org/10.1016/j.jbi.2013.12.006)

3. **Lee, J., Yoon, W., Kim, S., Kim, D., Kim, S., So, C. H., and Kang, J.** (2020). BioBERT: A pre-trained biomedical language representation model for biomedical text mining. *Bioinformatics*, 36(4), 1234–1240. [Link](https://doi.org/10.1093/bioinformatics/btz682)

4. **Li, J., Sun, Y., Johnson, R. J., Sciaky, D., Wei, C.-H., Leaman, R., Davis, A. P., Mattingly, C. J., Wiegers, T. C., and Lu, Z.** (2016). BioCreative V CDR task corpus: a resource for chemical disease relation extraction. *Database*, 2016. [Link](https://doi.org/10.1093/database/baw068)

5. **Nori, H., King, N., McKinney, S. M., Carignan, D., and Horvitz, E.** (2023). Capabilities of GPT-4 on medical challenge problems. [Link](https://arxiv.org/abs/2303.13375)

6. **Wang, Z., Zhao, K., Wang, Z., and Shang, J.** (2022). Formulating few-shot fine-tuning towards language model pre-training: A pilot study on named entity recognition. *Findings of the Association for Computational Linguistics: EMNLP 2022*, 3186–3199. [Link](https://doi.org/10.18653/v1/2022.findings-emnlp.233)

7. **Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi, E., Le, Q. V., Zhou, D., et al.** (2022). Chain-of-thought prompting elicits reasoning in large language models. *Advances in Neural Information Processing Systems*, 35, 24824–24837. [Link](https://arxiv.org/abs/2201.11903)


<hr>
<p><em>Note: The cover image/logo was created using OpenAI's DALL·E.</em></p>
