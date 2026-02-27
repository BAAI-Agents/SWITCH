# SWITCH: Benchmarking Modeling and Handling of Tangible Interfaces in Long-horizon Embodied Scenarios

<div align="center">

[[arXiv]](https://arxiv.org/abs/2511.17649)
[[leaderboard]](https://huggingface.co/spaces/BAAI-Agents/SWITCH-Basic-Leaderboard)
[[dataset]](https://huggingface.co/datasets/BAAI-Agents/SWITCH-Basic-v1-open)
[[PDF]](https://arxiv.org/pdf/2511.17649)

</div>

Autonomous intelligence requires not only perception and reasoning, but critically, effective interaction with the existing world and its infrastructure. Everyday environments are rich in tangible control interfaces (TCIs), e.g., light switches, appliance panels, and embedded GUIs, that demand commonsense and physics reasoning, but also causal prediction and outcome verification in time and space (e.g., delayed heating, remote lights). Moreover, failures here have potential safety implications, yet current benchmarks rarely test grounding, partial observability (video), or post-hoc verification in situated settings. We introduce SWITCH (Semantic World Interface Tasks for Control and Handling), an embodied, task-driven benchmark created through iterative releases to probe these gaps.

<div align="center">

![Overview](assets/Overview.png)

</div>

The figure above provides an overview of the SWITCH benchmark, using the case *Turn off all the lights* as a running example. SWITCH covers the collection and annotation of real-world TCI interaction data (*Collected Data*), which we systematically structure into five distinct tasks. These tasks are designed to evaluate models across three crucial capability dimensions: Perception/Spatial Reasoning, Causal Reasoning/Planning, and Verification. 

Furthermore, we leverage the benchmark to evaluate advanced generative models, like Veo3. By comparing generated videos against ground truth, we illustrate how current models still exhibit significant room for improvement in logical consistency and fine-grained interaction for real-word use, thus underscoring the importance of SWITCH's target scenarios.

## Key Tasks

- **Task-Aware Visual Question Answering (VQA)**: defines a task-aware VQA setting composed of two complementary sub-tasks: (a) *UI State Recognition*: assessing whether the model can recognize and describe the current state of TCI elements within the scene; and (b) *Goal-Oriented Reasoning*: testing whether the model can interpret the purpose and outcome of actions observed in dynamic video sequences, reasoning about whether these interactions successfully achieve the intended task goals. Together, these sub-tasks measure a model's ability to ground semantics in visual context, comprehend high-level task objectives, and reason about the operational consistency between actions and outcomes.

- **Semantic UI Comprehension**: tests whether a model can accurately localize and interpret actionable UI elements in cluttered or dynamic settings, reasoning about their spatial and functional relationships while inferring human intent. 

- **Action Generation**: evaluates a model's ability to infer intent and plan executable, context-aware action sequences that achieve user objectives. (a) *UI Action Identification*: Detect the relevant interaction region, recognize its affordances, and predict the appropriate mode of interaction; (b) *Action Execution Planning*: Generate the necessary physical actions to perform the intended TCI operation, such as moving toward the interface or manipulating its components.

- **State Transition Prediction**: evaluates a model's capacity for causal reasoning, short-term prediction, and fine-grained understanding of action–state dynamics in interactive environments. (a) *UI-State Transition*: Predict changes in the visual or functional state of TCI elements after an action (e.g., switch toggled, button activated). (b) *Environment-State Transition*: Predict corresponding updates in the surrounding physical or visual environment (e.g., lights turning on, revealing a new scene or perspective).
(c) *Coupled Transition*: Reason about interdependent updates where TCI and environment states jointly change as part of the same causal event (e.g., pressing a control panel triggers a screen update and changes ambient light level).

- **Result Verification**: (a) *Verification Planning*: Test whether the model can infer what actions or checks are required to verify the outcome of a previous operation, such as observing key indicators, querying system feedback, or performing follow-up interactions. (b) *Expected State Prediction*: Assess whether the model can predict what the expected state should look like after a successful interaction, providing a causal grounding for evaluating success or failure.

## Benchmark Highlights

![Details](assets/CaseStudy.png)
Case studies of representative errors in video generation. Examples using Veo3. Cropped image frames to focus on issues.

## Getting Started

The [SWITCH Benchmark Leaderboard](https://huggingface.co/spaces/BAAI-Agents/SWITCH-Basic-Leaderboard) is now hosted on HuggingFace and it includes instructions on how to run on the full-set evaluation.

## Contributing

We welcome contributions and feedback! Please feel free to submit issues or pull requests.

## Citation

If you utilize SWITCH scenarios or data in your research, please cite:

```bibtex
@article{switch2025,
  title={{SWITCH}: {B}enchmarking Modeling and Handling of Tangible Interfaces in Long-horizon Embodied Scenarios},
  author={Jieru Lin, Zhiwei Yu, Börje F. Karlsson},
  journal={arXiv preprint arXiv:2511.17649},
  year={2025}
}
```

## Contact

For questions or inquiries, please reach out to the BAAI-Agents team at ```baai-agents at baai.ac.cn```.
