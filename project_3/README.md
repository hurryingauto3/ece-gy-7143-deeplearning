Okay, this looks like a solid and interesting final project! Jailbreaking deep models is a very relevant topic in deep learning security. Let's break down the prompt and build a plan of attack to maximize your chances of getting a full grade.

**Goal:** Successfully attack a pre-trained ResNet-34 classifier on a subset of ImageNet, compare different attack types (L∞, L0 patch), and analyze transferability to another model. Deliver a high-quality report and reproducible codebase.

**Key Constraints & Requirements:**

*   **Group Size:** At most 3 people.
*   **Deadline:** May 12, 2025, 11:59 pm. (You have ample time, start early!)
*   **Grading:** 25% of overall grade. Focus on report quality, results quality, and code quality.
*   **Deliverables:** PDF Report (AAAI 2024 format, <=4 pages main + appendix, GitHub link on page 1), Public GitHub Repo (reproducible code, notebooks, clear output).
*   **Models:** ResNet-34 (target for attacks), another TorchVision ImageNet model (for transferability).
*   **Data:** Provided test dataset (subset of ImageNet-1K, 100 classes, 500 images).
*   **Preprocessing:** Standard ImageNet normalization.
*   **Metrics:** Top-1 and Top-5 accuracy.
*   **Attacks:**
    *   Task 2: FGSM (L∞, ε=0.02). Target: >50% relative accuracy drop.
    *   Task 3: Improved L∞ (e.g., PGD, targeted). L∞, ε=0.02. Target: >70% relative accuracy drop.
    *   Task 4: Patch Attack (L0 constraint, L∞ within patch). Random 32x32 patch, larger ε (0.3-0.5). Targeted hint.
*   **Visualizations:** 3-5 examples for Tasks 2, 3, 4 (original, perturbed, predictions). Include in appendix.
*   **Analysis:** Compare attack effectiveness, discuss L∞ vs L0 patch, analyze transferability (Task 5 findings).
*   **Citations:** Cite external code, LLMs, resources.

**Recommended Plan of Attack:**

Break this down into phases and assign tasks within your group. **Start Early!** The report formatting and debugging adversarial attacks can take significant time.

**Phase 1: Setup and Baseline (Estimated Time: 1-2 days)**

1.  **Form Your Group:** If you haven't already, finalize your team (max 3).
2.  **Set up GitHub Repository:** Create a *public* repository on GitHub. Add all team members as collaborators. Structure it nicely (e.g., `src/`, `notebooks/`, `reports/`).
3.  **Clone Repo & Set up Environment:**
    *   Everyone clones the repository.
    *   Install necessary libraries: `pytorch`, `torchvision`, `numpy`, `matplotlib`, `Pillow`, `json`, etc. Consider using a virtual environment (`conda` or `venv`).
    *   *Recommendation:* If you have access to NYU HPC or Google Colab Pro, set that up *now*. Running these attacks, especially multi-step ones, on a CPU will be very slow.
4.  **Download Data and Model:**
    *   Download the provided test dataset (`TestDataSet` folder and `.json` file). Place it in your project structure.
    *   Write code to load the ResNet-34 model: `pretrained_model = torchvision.models.resnet34(weights='IMAGENET1K_V1')`. Move the model to GPU if available (`model.to(device)`). Set the model to evaluation mode (`model.eval()`).
5.  **Implement Data Loading and Preprocessing:**
    *   Write code using `torchvision.datasets.ImageFolder` and the provided `plain_transforms`.
    *   Implement a PyTorch `DataLoader` for batching (helpful for efficiency).
6.  **Implement Evaluation Function:** Write a function that takes a model and a DataLoader, calculates Top-1 and Top-5 accuracy, given the label index mapping from the JSON file.
7.  **Run Task 1 (Baseline):**
    *   Evaluate the pre-trained ResNet-34 on the *original* test dataset.
    *   **Crucially:** Print the Top-1 and Top-5 accuracy clearly. Save these numbers! This is your baseline.
8.  **Code Structure & Notebooks:** Create a Jupyter notebook or script (`task1_baseline.ipynb` or similar) that demonstrates these steps and prints the results. Make sure it's clear and runnable.

**Phase 2: Implement FGSM (Task 2) (Estimated Time: 2-3 days)**

1.  **Understand FGSM:** Review the formula `x <- x + ε sign (∇xL)`. Understand that you need to:
    *   Compute the loss (`torch.nn.CrossEntropyLoss` is typical) for the original image and true label.
    *   Compute the gradient of the loss *with respect to the input image tensor*. (`image_tensor.requires_grad = True`, `loss.backward()`, then access `image_tensor.grad`). Remember to zero gradients before computing them (`model.zero_grad()` and potentially `image_tensor.grad.zero_()`, or ensure gradients are computed fresh per image/batch).
    *   Get the sign of the gradient (`torch.sign()`).
    *   Multiply by ε.
    *   Add the perturbation to the original image tensor.
    *   **Important:** Handle normalization and clipping. The perturbation is added to the *normalized* tensor. After adding, you must clip the *perturbed* tensor's values to the valid range (e.g., 0-1 if that's the normalized range). You might also want to ensure the perturbation itself (the difference between original and perturbed) does not exceed ε *after* clipping the result, although the standard FGSM formula implies adding the step then clipping the output.
2.  **Implement FGSM Attack Function:** Write a function `fgsm_attack(model, image, label, epsilon)` that returns the adversarial image tensor.
3.  **Generate Adversarial Test Set 1:**
    *   Iterate through the *original* test dataset DataLoader.
    *   For each image and label, call your `fgsm_attack` function with ε=0.02.
    *   Save the *perturbed* images. **How to save?** The easiest way for later use with `ImageFolder` and `plain_transforms` is to save the perturbed *tensors* as image files. Remember to *unnormalize* before saving to typical image formats (like PNG or JPEG) where pixel values are 0-255. Make sure your loading process with `plain_transforms` correctly reverses this. *Alternative (and potentially safer):* Save the perturbed images in a structure identical to the original `TestDataSet` folder structure, then reload them using `ImageFolder` and `plain_transforms` for evaluation. This ensures the evaluation uses the exact tensor representation you saved.
4.  **Verify Constraint & Visualize:**
    *   After generating, load a few perturbed images.
    *   Calculate the L∞ distance between the loaded perturbed image tensor and the original image tensor. Verify it's <= ε (allowing for floating point precision).
    *   Visualize 3-5 examples. Show the original image, its predicted label (Top-1), the perturbed image, and its new predicted label (Top-1 or Top-5). Note if the prediction changed and if it's now incorrect. Include the L∞ distance for visual examples.
5.  **Evaluate Task 2:**
    *   Load Adversarial Test Set 1 using `ImageFolder` and `plain_transforms`.
    *   Evaluate ResNet-34 on this new dataset using your evaluation function.
    *   **Crucially:** Print the new Top-1 and Top-5 accuracy. Check if you met the >50% relative drop target from Task 1.
6.  **Code Structure & Notebook:** Create `task2_fgsm.ipynb` showing the implementation, generation, verification, visualization, and evaluation steps with clear outputs.

**Phase 3: Implement Improved L∞ Attack (Task 3) (Estimated Time: 3-4 days)**

1.  **Research/Choose Attack:**
    *   **Strong Recommendation:** Implement Projected Gradient Descent (PGD). It's an iterative version of FGSM and is a standard benchmark. It's relatively easy to implement if you have FGSM working.
    *   *Other options:* Targeted attacks (can be combined with FGSM/PGD), C&W (more complex, potentially different distance metric). Stick to PGD for simplicity and effectiveness within the L∞ constraint.
    *   **PGD:** It performs multiple small gradient steps, clipping the perturbation after *each* step back into the L∞ ball of radius ε centered around the *original* image. It often adds random initialization within the ε ball.
2.  **Implement Improved Attack Function:** Write a function `improved_attack(model, image, label, epsilon, num_steps, step_size, random_init)` for PGD or your chosen method.
3.  **Hyperparameter Tuning:** Experiment with `num_steps` and `step_size` for PGD. A common setup is `step_size = epsilon / num_steps` or slightly larger, but tune to see what works best while staying within the epsilon budget.
4.  **Generate Adversarial Test Set 2:**
    *   Use your improved attack function on the original dataset with ε=0.02.
    *   Save the perturbed images using the same method as in Task 2.
5.  **Verify Constraint & Visualize:**
    *   Load and verify the L∞ distance for Adversarial Test Set 2 (should be <= ε=0.02).
    *   Visualize 3-5 examples as in Task 2.
6.  **Evaluate Task 3:**
    *   Load Adversarial Test Set 2.
    *   Evaluate ResNet-34.
    *   **Crucially:** Print the new Top-1 and Top-5 accuracy. Check if you met the >70% relative drop target. If not, revisit hyperparameter tuning or consider a targeted variant.
7.  **Code Structure & Notebook:** Create `task3_improved.ipynb` detailing the implementation, hyperparameter choices (and why), generation, verification, visualization, and evaluation.

**Phase 4: Implement Patch Attack (Task 4) (Estimated Time: 3-4 days)**

1.  **Choose Base Method:** Pick your best performing L∞ method from Task 3 (likely PGD) or FGSM.
2.  **Modify for Patch:**
    *   Select a random 32x32 pixel patch for each image. Where is the patch located? (Could be completely random location or random starting corner). Random starting corner seems reasonable.
    *   Compute the gradient *only* within this patch region of the input tensor. Or compute the full gradient and then zero out the gradient outside the patch before taking the sign step.
    *   Apply the perturbation *only* to the pixels within the selected patch.
    *   Allow a larger ε (0.3 or 0.5) *within the patch*. This means the L∞ distance constraint applies *only* to the pixels inside the patch, but with a larger budget. Pixels outside the patch have an L∞ distance of 0.
    *   **Hint: Targeted Attack:** Why might a targeted attack be helpful? With fewer pixels to twiddle, aiming for a specific wrong class might be more effective than just trying to make the model uncertain or predict *any* wrong class. Consider implementing a targeted variant where you try to make the model predict a *random wrong class* or a *fixed target class* (e.g., "toaster" or "ostrich"). A simple target could be the class index that maximizes the gradient within the patch, shifted away from the true class. Or simply pick a random class index different from the true one.
3.  **Implement Patch Attack Function:** Write `patch_attack(model, image, label, patch_size, epsilon, num_steps, step_size, targeted=False, target_class=None)`.
4.  **Hyperparameter Tuning:** Experiment with ε (0.3, 0.5, maybe higher?) and potentially `num_steps`/`step_size` if using an iterative patch attack.
5.  **Generate Adversarial Test Set 3:**
    *   Use your patch attack function on the original dataset.
    *   Save the perturbed images (remembering that only a patch is changed).
6.  **Verify Constraint & Visualize:**
    *   Load the images. The L∞ distance *globally* will likely exceed 0.02, but verify that *outside* the patch, the image is unchanged, and *inside* the patch, the L∞ perturbation is within the larger ε budget.
    *   Visualize 3-5 examples. Clearly show the original, the perturbed image with the visible patch perturbation, and the predictions. You might want to highlight the patch location (e.g., with a bounding box) for clarity in the visualization appendix.
7.  **Evaluate Task 4:**
    *   Load Adversarial Test Set 3.
    *   Evaluate ResNet-34.
    *   **Crucially:** Print the new Top-1 and Top-5 accuracy. This will likely be higher than Task 2/3 because the attack is more constrained (L0).
8.  **Code Structure & Notebook:** Create `task4_patch.ipynb` detailing the implementation (including patch selection), hyperparameter choices, generation, verification, visualization, and evaluation. Discuss challenges faced due to the L0 constraint.

**Phase 5: Transferability Analysis (Task 5) (Estimated Time: 1-2 days)**

1.  **Load Second Model:** Choose another ImageNet-1K model from TorchVision (e.g., DenseNet-121, VGG-16, MobileNetV2, EfficientNet). Load it and move it to GPU. Set it to evaluation mode.
2.  **Evaluate All Datasets:** Use your existing evaluation function to evaluate this *new* model on:
    *   Original Test Dataset
    *   Adversarial Test Set 1 (FGSM)
    *   Adversarial Test Set 2 (Improved L∞)
    *   Adversarial Test Set 3 (Patch)
3.  **Collect Results:** Create a table summarizing Top-1 and Top-5 accuracy for *both* models across *all four* datasets. This table will be essential for your report.
4.  **Analysis:**
    *   Compare ResNet-34 performance across the datasets (Original > Adv 1 > Adv 2 > Adv 3 - likely trend).
    *   Compare the second model's performance across the datasets.
    *   *Key Question:* How well do the attacks *transfer*? Does an attack effective against ResNet-34 also significantly degrade performance on the other model? Which attack type transfers best? (L∞ attacks, especially PGD, often transfer better than L0 or highly-tuned targeted attacks).
    *   Discuss *why* some attacks transfer better than others (e.g., reliance on model-specific gradients vs. capturing more general vulnerabilities or features).
    *   Discuss potential ways to *mitigate* transferability or defend against these attacks (e.g., adversarial training, ensemble methods, defensive distillation - briefly mention these concepts if you encountered them in class or research).

**Phase 6: Report Writing and Codebase Finalization (Estimated Time: 3-5 days)**

1.  **Get the AAAI 2024 Template:** Download the "Camera-ready" template (LaTeX or Word) ASAP. Start putting your content into it immediately. Don't wait until the end.
2.  **Draft Report Sections:**
    *   **Title/Authors:** Add project title, team member names.
    *   **GitHub Link:** Put the link to your public repo prominently on the first page.
    *   **Overview:** Write a concise summary of your approach and key findings (e.g., which attack was most effective, how well attacks transferred).
    *   **Methodology:** Detail *how* you implemented each task. Explain FGSM, your chosen improved L∞ attack (PGD?), and your patch attack implementation (patch selection, how perturbation applied, larger ε). Discuss hyperparameter choices and tuning process. Mention challenges and lessons learned for each task. Explain the evaluation setup and transferability methodology.
    *   **Results:** Present your accuracy table clearly (Model vs. Dataset for Top-1 and Top-5). Report the perturbation sizes (ε used for L∞, patch size and ε for L0). Briefly mention training/attack times if notable (e.g., FGSM is fast, PGD is slower, patch attack time depends on steps). Refer to visualizations in the appendix.
    *   **Analysis/Discussion:** Expand on the findings from Task 5 transferability. Discuss the effectiveness of L∞ vs. L0 patch attacks. Interpret the results from your table.
    *   **Citations:** Add a bibliography section for any resources, papers, code snippets, or LLMs you used.
    *   **Appendix:** Include your 3-5 visualizations for Tasks 2, 3, and 4. Add any extra plots or analysis that didn't fit in the 4 pages.
3.  **Refine Report:** Edit for clarity, conciseness, and flow. Ensure it fits within the 4-page limit for the main content. Check formatting carefully against the AAAI template. Proofread for typos and grammatical errors.
4.  **Finalize Codebase:**
    *   Clean up your notebooks/scripts. Ensure they are well-commented.
    *   Add a `README.md` to your GitHub repo explaining how to run your code, reproduce the results, and outlining the project structure.
    *   Make sure the code is truly reproducible (e.g., data paths are relative or configurable, necessary libraries are listed).
    *   Ensure all required output (especially the accuracy numbers) is printed clearly in your notebooks or scripts.
5.  **Final Review:** Have everyone on the team review the report and codebase one last time against the project requirements and grading criteria.
6.  **Submission:** One team member uploads the final PDF report to Gradescope and *tags all other team members*. Ensure the GitHub repo is public and the link in the report is correct.

**Tips for Success:**

*   **Start with Task 1 (Baseline):** This is non-negotiable. You *must* get the baseline evaluation working correctly before attempting attacks.
*   **Implement Incrementally:** Get FGSM working first. Debugging adversarial attacks can be tricky (gradients, normalization, clipping). Once FGSM is solid, modifying it for PGD is much easier. Modifying for the patch attack is the next step.
*   **Visualize Early and Often:** Visualizing the perturbed images and predictions is crucial for debugging. Does the perturbation look right? Did the prediction change as expected?
*   **Verify Constraints:** Explicitly calculate and print the L∞ distance for a few examples to confirm you are meeting the ε budget in Tasks 2 & 3 (and the patch constraints in Task 4).
*   **Use GPU:** Seriously, use a GPU. Attacks involving gradient calculations are much faster.
*   **Batching:** Process images in batches for efficiency.
*   **Handle Normalization:** Be very careful about when you normalize/unnormalize images, and whether ε applies to the 0-1, 0-255, or normalized tensor range. The prompt implies ε=0.02 applies to the normalized range after the standard transforms.
*   **Report Formatting is Key:** Don't underestimate the time needed for the AAAI template. Start early.
*   **Collaborate Effectively:** Use GitHub for version control. Divide tasks but also code review each other's work. Discuss challenges together.
*   **Utilize Resources:** Refer to the provided links, class notes, PyTorch documentation, and relevant research papers if you choose more advanced attacks. *Cite everything you use.*

By following this structured plan, dividing the work, and starting early, you should be well-equipped to tackle this project successfully and achieve a full grade. Good luck!