# Deploy LLM to Edge Device
A browser-based AI chatbot powered by WebLLM that runs W4A16 quantized Llama-3.2-3B-Instruct model locally using WebGPU.

## How to Run

1. **Navigate to the Project Directory**
```bash
git clone https://github.com/cloud-peterjohn/Deploy-LLM-on-Edge-Deivce.git
cd Deploy-LLM-on-Edge-Deivce
```

2. **Start a Local HTTP Server**
```bash
python -m http.server 8000
```

3. **Access in Browser**
Navigate to `http://localhost:8000/test.html` in a WebGPU-compatible browser (Chrome 113+, Firefox 110+, Safari 16.4+).

4. **Select Model**
Choose from available quantized models, including the W4A16 Llama-3.2-3B-Instruct model quantized by us. (highlighted in red)

5. **Download Model**
Click "Download" to cache the selected model locally (first download may take several minutes depending on model size)

6. **Start Chatting**
Once downloaded, interact with the model through the chat interface with real-time streaming responses. Throughput will also be displayed in the web page.

## File Structure

```
├── test.html          # Main HTML interface
├── test.css           # Styling and layout
├── test.js            # WebLLM logic and GPU monitoring
└── ReadMe.md          # This file
```

## Motivation
The rapid advancement of large language models has created a significant barrier for widespread adoption due to their substantial computational requirements and memory footprint. Most state-of-the-art models require expensive cloud infrastructure or high-end datacenter GPUs, limiting accessibility for individual users and edge applications. This project addresses these challenges by demonstrating how quantization techniques combined with optimized inference frameworks can enable powerful language models to run locally on consumer hardware.

We selected Llama-3.2-3B-Instruct as our target model due to its optimal balance between performance and computational efficiency. With 3 billion parameters, it provides sophisticated conversational capabilities while remaining small enough for aggressive quantization without severe quality degradation. The model's instruction-tuned nature ensures reliable chat performance, making it ideal for demonstrating practical edge deployment scenarios.

Our device selection encompasses both discrete GPU (NVIDIA RTX 3060) and integrated GPU (AMD Radeon 780M) to showcase the versatility of our approach across different hardware configurations. The RTX 3060 represents a mainstream gaming GPU accessible to many users, while the Radeon 780M demonstrates deployment on ultra-portable laptops without dedicated graphics cards. This diversity validates that quantized models can democratize AI access across various personal computing devices.

By leveraging AutoGPTQ quantization to compress weights to 4-bit precision and MLC-LLM for WebGPU optimization, we transform a multi-gigabyte model into a 2.3GB package that runs entirely in web browsers. This approach eliminates server dependencies, ensures data privacy through local processing, and reduces operational costs, making sophisticated AI capabilities accessible to individual users on their personal devices.

## Methods: 
### Quantization with AutoGPTQ
We use AutoGPTQ, a post-training weight-only quantization library, to quantize the Llama-3.2-3B-Instruct model to W4A16 format. 
AutoGPTQ is an implementation of the GPTQ (Generative Pre-trained Transformer Quantization) algorithm, which operates on the principle of optimal weight quantization through second-order optimization, leveraging the structure of transformer architectures to achieve superior compression-accuracy trade-offs. 
The core mathematical foundation of GPTQ rests upon the formulation of quantization as a layer-wise optimization problem. Given a weight matrix $\mathbf{W} \in \mathbb{R}^{d_{out} \times d_{in}}$ and its corresponding input activations $\mathbf{X} \in \mathbb{R}^{d_{in} \times n}$ collected from calibration data, the objective is to find a quantized weight matrix $\hat{\mathbf{W}}$ that minimizes the squared reconstruction error:
$$\mathcal{L} = ||\mathbf{W}\mathbf{X} - \hat{\mathbf{W}}\mathbf{X}||_F^2$$
where $||\cdot||_F$ denotes the Frobenius norm. This formulation ensures that the quantized layer produces outputs as close as possible to the original full-precision layer when presented with representative input data.

The innovation of GPTQ lies in its utilization of second-order information to guide the quantization process. The algorithm computes the Hessian matrix of the objective function with respect to the weights:
$$\mathbf{H} = \frac{2}{n}\mathbf{X}\mathbf{X}^T$$
This Hessian captures the curvature of the loss landscape, providing crucial information about parameter sensitivity. Parameters corresponding to directions of high curvature (large eigenvalues of $\mathbf{H}$) require more careful quantization, as small perturbations in these directions lead to significant increases in the reconstruction error.
The quantization procedure proceeds in a greedy, column-wise manner. For each column $\mathbf{w}_i$ of the weight matrix, GPTQ computes the optimal quantization by solving:
$$\hat{w}{i,j} = \text{Quantize}(w{i,j})$$
where the quantization function maps continuous weights to discrete levels. 

The quantization error for each weight is computed as:
$$e_{i,j} = w_{i,j} - \hat{w}_{i,j}$$
To compensate for this quantization error and prevent its accumulation across the network, GPTQ applies a correction mechanism. The key insight is that the error introduced by quantizing weight $w_{i,j}$ can be partially compensated by adjusting the remaining unquantized weights in the same row. The correction is computed using the inverse Hessian:
$$\mathbf{W}{i, j+1:d{in}} = \mathbf{W}{i, j+1:d{in}} - \frac{e_{i,j}}{\mathbf{H}{j,j}} \mathbf{H}{j, j+1:d_{in}}$$
This update rule ensures that the cumulative effect of quantization errors is minimized by optimally redistributing the error among subsequent weights according to the second-order structure of the problem.
However, the Hessian inverse computation presents computational challenges for large matrices. GPTQ addresses this through an efficient Cholesky decomposition approach, updating the Hessian inverse incrementally as weights are quantized. When quantizing weight $w_{i,j}$, the corresponding row and column of the Hessian inverse are updated using the Sherman-Morrison formula:
$$\mathbf{H}^{-1}{new} = \mathbf{H}^{-1}{old} - \frac{\mathbf{H}^{-1}_{old} \mathbf{e}_j \mathbf{e}j^T \mathbf{H}^{-1}{old}}{\mathbf{e}j^T \mathbf{H}^{-1}{old} \mathbf{e}_j}$$
where $\mathbf{e}_j$ is the $j$-th standard basis vector.

The quantization scheme employed in GPTQ typically utilizes uniform quantization with learned scales and zero points. For $b$-bit quantization, weights are mapped to integers in the range $[0, 2^b-1]$ according to:
$$\hat{w} = \text{clamp}\left(\left\lfloor\frac{w - z}{s}\right\rceil, 0, 2^b-1\right)$$
where $s$ is the scale factor, $z$ is the zero point, and $\lfloor\cdot\rceil$ denotes rounding to the nearest integer. The scale and zero point are computed to minimize quantization error within each group of weights.

Group-wise quantization further enhances the method's effectiveness by partitioning weight matrices into smaller groups, each with its own quantization parameters. This approach captures local statistics more effectively, particularly important for transformer architectures where different attention heads or feed-forward components may exhibit distinct weight distributions. The group size $g$ becomes a hyperparameter that trades off between quantization accuracy and metadata overhead.

Our quantization configuration for the Llama-3.2-3B-Instruct model is 4-bit W4A16, which means we quantize the weights to 4 bits. Moreover, we use group-wise quantization with a group size of 128, allowing us to capture local weight distributions effectively while maintaining a manageable metadata overhead. After quantization as in [code](https://github.com/cloud-peterjohn/LLM-Acceleration/blob/main/gptq-quant.ipynb), we push the quantized model to [HuggingFace](zbyzby/Llama-3.2-1B-Instruct-GPTQ-Quant).

### Transform into MLC-LLM Model
We use the MLC-LLM library to convert the quantized model into specific formats, and you can install it [here](https://llm.mlc.ai/docs/install/mlc_llm.html#install-mlc-packages).
Then verify installation by running:
```bash
mlc_llm --help
```
Next, we install TVM Unity Compiler, you can install it [here](https://llm.mlc.ai/docs/install/tvm.html#install-tvm-unity)
Then verify installation by running:
```bash
python -c "import tvm; print(tvm.__file__)"
```
Now we can clone model from HuggingFace and convert weight.
```bash
mkdir -p dist/models && cd dist/models
git lfs install
git clone https://huggingface.co/zbyzby/Llama-3.2-1B-Instruct-GPTQ-Quant
cd ../..
mlc_llm convert_weight ./dist/models/Llama-3.2-1B-Instruct/ --quantization q4f16_1 -o dist/Llama-3.2-3B-Instruct-Quantized-for-MLC
```
Then we use `gen_config` to generate mlc-chat-config.json and process tokenizers. 
```bash
mlc_llm gen_config ./dist/models/Llama-3.2-1B-Instruct/ --quantization q4f16_1 --conv-template wizard_coder_or_math -o dist/Llama-3.2-3B-Instruct-Quantized-for-MLC/
```
Finally, we upload the model to [HuggingFace](https://huggingface.co/zbyzby/Llama-3.2-3B-Instruct-Quantized-for-MLC).

### Run Locally

We implements a browser-based AI chatbot that runs large language models entirely on the user side using WebGPU technology. 

The frontend consists of three main components:

- **HTML Interface (`test.html`)**: Provides a clean, responsive UI with model selection dropdown, download progress tracking, chat interface, and real-time GPU monitoring dashboard
- **CSS Styling (`test.css`)**: Implements a modern purple-themed design with responsive layout, message bubbles, and GPU information display panels
- **JavaScript Logic (`test.js`)**: Handles WebLLM integration, model management, chat functionality, and GPU resource monitoring using the WebLLM library


The application leverages WebLLM to run language models directly in the browser:

1. **Model Download & Caching**: Models are downloaded once and cached in browser storage (IndexedDB) for persistent local access
2. **WebGPU Acceleration**: Utilizes the user's GPU through WebGPU API for efficient model inference, enabling real-time chat responses
3. **Memory Management**: Automatically manages GPU memory allocation and monitors VRAM usage to ensure stable performance
4. **Quantized Models**: Supports various quantization formats (4-bit, 16-bit) to balance model quality with resource requirements

## Results
Before quantization, the pre-finetuned Llama-3.2-3B-Instruct model has a perplexity of **9.5** on the WikiText2 dataset, while the quantized W4A16 model achieves a perplexity of **11.21**.

We deployed the quantized model on **NVIDIA RTX 3060** GPU in laptop, as shown in this [video](https://github.com/cloud-peterjohn/Deploy-LLM-on-Edge-Deivce/blob/main/images/Recording.mp4). 
The NVIDIA GeForce RTX 3060, built on the Ampere architecture with a GA106 core, features 3584 CUDA cores, a boost clock of 1777 MHz, 6GB VRAM memory, supports DLSS 2.0 and second-gen ray tracing, and delivers strong 1080p and 1440p gaming performance while maintaining a 170W TDP.
On this GPU, the model can run with a throughput of **25.1744** tokens per second.
![3060-tput](https://github.com/cloud-peterjohn/Deploy-LLM-on-Edge-Deivce/blob/main/images/NVIDIA-RTX3060.png)

Moreover, we deploy the quantized model on **AMD 7840H** in laptop. 
The AMD Ryzen 7 7840H features an integrated GPU known as the Radeon 780M, which features RDNA 3 architecture with 12 Compute Units (768 stream processors), and supports DirectX 12 Ultimate, Vulkan, and basic ray tracing. 
Notably, the Radeon 780M integrated GPU in the AMD 7840H does not have dedicated VRAM but dynamically shares system memory (RAM), with allocation varying based on BIOS settings, OS management, and the installed RAM type and capacity. 
On this GPU, the model can run with a throughput of **11.1195** tokens per second. 
![7840-tput](https://github.com/cloud-peterjohn/Deploy-LLM-on-Edge-Deivce/blob/main/images/Radeon-780M.png)

## References
- [AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ.git): https://github.com/AutoGPTQ/AutoGPTQ.git
- [GPTQ-for-LLaMa](https://github.com/qwopqwop200/GPTQ-for-LLaMa.git): https://github.com/qwopqwop200/GPTQ-for-LLaMa.git
- [GPTQ](https://github.com/IST-DASLab/gptq.git): https://github.com/IST-DASLab/gptq.git
- [MLC-LLM](https://llm.mlc.ai/docs/): https://github.com/mlc-ai/mlc-llm.git
- [Web-LLM](https://webllm.mlc.ai/docs/): https://github.com/mlc-ai/web-llm.git


