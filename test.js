import * as webllm from "https://esm.run/@mlc-ai/web-llm";

/*************** WebLLM logic ***************/
const modelVersion = "v0_2_48";
const modelLibURLPrefix = "https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/web-llm-models/";
const messages = [
    {
        content: "You are a helpful AI agent helping users.",
        role: "system",
    },
];

const myAppConfig = {
    model_list: [
        {
            model: "https://huggingface.co/zbyzby/Llama-3.2-3B-Instruct-Quantized-for-MLC",
            model_id: "Llama-3.2-3B-Instruct-W4A16-Quantized",
            model_lib:
                modelLibURLPrefix +
                modelVersion +
                "/Llama-3.2-3B-Instruct-q4f16_1-ctx4k_cs1k-webgpu.wasm",
            vram_required_MB: 2263.69,
            low_resource_required: true,
            overrides: {
                context_window_size: 4096,
            },
        },
        {
            model: "https://huggingface.co/mlc-ai/Llama-3.2-1B-Instruct-q4f32_1-MLC",
            model_id: "Llama-3.2-1B-Instruct-q4f32_1-MLC",
            model_lib:
                modelLibURLPrefix +
                modelVersion +
                "/Llama-3.2-1B-Instruct-q4f32_1-ctx4k_cs1k-webgpu.wasm",
            vram_required_MB: 1128.82,
            low_resource_required: true,
            overrides: {
                context_window_size: 4096,
            },
        },
        {
            model: "https://huggingface.co/mlc-ai/Llama-3.2-1B-Instruct-q4f16_1-MLC",
            model_id: "Llama-3.2-1B-Instruct-q4f16_1-MLC",
            model_lib:
                modelLibURLPrefix +
                modelVersion +
                "/Llama-3.2-1B-Instruct-q4f16_1-ctx4k_cs1k-webgpu.wasm",
            vram_required_MB: 879.04,
            low_resource_required: true,
            overrides: {
                context_window_size: 4096,
            },
        },
        {
            model: "https://huggingface.co/mlc-ai/Llama-3.2-1B-Instruct-q0f32-MLC",
            model_id: "Llama-3.2-1B-Instruct-q0f32-MLC",
            model_lib:
                modelLibURLPrefix +
                modelVersion +
                "/Llama-3.2-1B-Instruct-q0f32-ctx4k_cs1k-webgpu.wasm",
            vram_required_MB: 5106.26,
            low_resource_required: true,
            overrides: {
                context_window_size: 4096,
            },
        },
        {
            model: "https://huggingface.co/mlc-ai/Llama-3.2-1B-Instruct-q0f16-MLC",
            model_id: "Llama-3.2-1B-Instruct-q0f16-MLC",
            model_lib:
                modelLibURLPrefix +
                modelVersion +
                "/Llama-3.2-1B-Instruct-q0f16-ctx4k_cs1k-webgpu.wasm",
            vram_required_MB: 2573.13,
            low_resource_required: true,
            overrides: {
                context_window_size: 4096,
            },
        },
        {
            model: "https://huggingface.co/mlc-ai/Llama-3.1-8B-Instruct-q4f32_1-MLC",
            model_id: "Llama-3.1-8B-Instruct-q4f32_1-MLC-1k",
            model_lib:
                modelLibURLPrefix +
                modelVersion +
                "/Llama-3_1-8B-Instruct-q4f32_1-ctx4k_cs1k-webgpu.wasm",
            vram_required_MB: 5295.7,
            low_resource_required: true,
            overrides: {
                context_window_size: 1024,
            },
        },
        {
            model: "https://huggingface.co/mlc-ai/Llama-3.1-8B-Instruct-q4f16_1-MLC",
            model_id: "Llama-3.1-8B-Instruct-q4f16_1-MLC-1k",
            model_lib:
                modelLibURLPrefix +
                modelVersion +
                "/Llama-3_1-8B-Instruct-q4f16_1-ctx4k_cs1k-webgpu.wasm",
            vram_required_MB: 4598.34,
            low_resource_required: true,
            overrides: {
                context_window_size: 1024,
            },
        },
        {
            model: "https://huggingface.co/mlc-ai/Llama-3.1-8B-Instruct-q4f32_1-MLC",
            model_id: "Llama-3.1-8B-Instruct-q4f32_1-MLC",
            model_lib:
                modelLibURLPrefix +
                modelVersion +
                "/Llama-3_1-8B-Instruct-q4f32_1-ctx4k_cs1k-webgpu.wasm",
            vram_required_MB: 6101.01,
            low_resource_required: false,
            overrides: {
                context_window_size: 4096,
            },
        },
        {
            model: "https://huggingface.co/mlc-ai/Llama-3.1-8B-Instruct-q4f16_1-MLC",
            model_id: "Llama-3.1-8B-Instruct-q4f16_1-MLC",
            model_lib:
                modelLibURLPrefix +
                modelVersion +
                "/Llama-3_1-8B-Instruct-q4f16_1-ctx4k_cs1k-webgpu.wasm",
            vram_required_MB: 5001.0,
            low_resource_required: false,
            overrides: {
                context_window_size: 4096,
            },
        },
        {
            model:
                "https://huggingface.co/mlc-ai/DeepSeek-R1-Distill-Qwen-7B-q4f16_1-MLC",
            model_id: "DeepSeek-R1-Distill-Qwen-7B-q4f16_1-MLC",
            model_lib:
                modelLibURLPrefix +
                modelVersion +
                "/Qwen2-7B-Instruct-q4f16_1-ctx4k_cs1k-webgpu.wasm",
            low_resource_required: false,
            vram_required_MB: 5106.67,
            overrides: {
                context_window_size: 4096,
            },
        },
        {
            model:
                "https://huggingface.co/mlc-ai/DeepSeek-R1-Distill-Qwen-7B-q4f32_1-MLC",
            model_id: "DeepSeek-R1-Distill-Qwen-7B-q4f32_1-MLC",
            model_lib:
                modelLibURLPrefix +
                modelVersion +
                "/Qwen2-7B-Instruct-q4f32_1-ctx4k_cs1k-webgpu.wasm",
            low_resource_required: false,
            vram_required_MB: 5900.09,
            overrides: {
                context_window_size: 4096,
            },
        },
        {
            model:
                "https://huggingface.co/mlc-ai/DeepSeek-R1-Distill-Llama-8B-q4f32_1-MLC",
            model_id: "DeepSeek-R1-Distill-Llama-8B-q4f32_1-MLC",
            model_lib:
                modelLibURLPrefix +
                modelVersion +
                "/Llama-3_1-8B-Instruct-q4f32_1-ctx4k_cs1k-webgpu.wasm",
            vram_required_MB: 6101.01,
            low_resource_required: false,
            overrides: {
                context_window_size: 4096,
            },
        },
        {
            model:
                "https://huggingface.co/mlc-ai/DeepSeek-R1-Distill-Llama-8B-q4f16_1-MLC",
            model_id: "DeepSeek-R1-Distill-Llama-8B-q4f16_1-MLC",
            model_lib:
                modelLibURLPrefix +
                modelVersion +
                "/Llama-3_1-8B-Instruct-q4f16_1-ctx4k_cs1k-webgpu.wasm",
            vram_required_MB: 5001.0,
            low_resource_required: false,
            overrides: {
                context_window_size: 4096,
            },
        },
        {
            model:
                "https://huggingface.co/mlc-ai/Hermes-2-Theta-Llama-3-8B-q4f16_1-MLC",
            model_id: "Hermes-2-Theta-Llama-3-8B-q4f16_1-MLC",
            model_lib:
                modelLibURLPrefix +
                modelVersion +
                "/Llama-3-8B-Instruct-q4f16_1-ctx4k_cs1k-webgpu.wasm",
            vram_required_MB: 4976.13,
            low_resource_required: false,
            overrides: {
                context_window_size: 4096,
            },
        },
        {
            model:
                "https://huggingface.co/mlc-ai/Hermes-2-Theta-Llama-3-8B-q4f32_1-MLC",
            model_id: "Hermes-2-Theta-Llama-3-8B-q4f32_1-MLC",
            model_lib:
                modelLibURLPrefix +
                modelVersion +
                "/Llama-3-8B-Instruct-q4f32_1-ctx4k_cs1k-webgpu.wasm",
            vram_required_MB: 6051.27,
            low_resource_required: false,
            overrides: {
                context_window_size: 4096,
            },
        },
        {
            model:
                "https://huggingface.co/mlc-ai/Hermes-2-Pro-Llama-3-8B-q4f16_1-MLC",
            model_id: "Hermes-2-Pro-Llama-3-8B-q4f16_1-MLC",
            model_lib:
                modelLibURLPrefix +
                modelVersion +
                "/Llama-3-8B-Instruct-q4f16_1-ctx4k_cs1k-webgpu.wasm",
            vram_required_MB: 4976.13,
            low_resource_required: false,
            overrides: {
                context_window_size: 4096,
            },
        },
        {
            model:
                "https://huggingface.co/mlc-ai/Hermes-2-Pro-Llama-3-8B-q4f32_1-MLC",
            model_id: "Hermes-2-Pro-Llama-3-8B-q4f32_1-MLC",
            model_lib:
                modelLibURLPrefix +
                modelVersion +
                "/Llama-3-8B-Instruct-q4f32_1-ctx4k_cs1k-webgpu.wasm",
            vram_required_MB: 6051.27,
            low_resource_required: false,
            overrides: {
                context_window_size: 4096,
            },
        },
        {
            model: "https://huggingface.co/mlc-ai/Hermes-3-Llama-3.2-3B-q4f32_1-MLC",
            model_id: "Hermes-3-Llama-3.2-3B-q4f32_1-MLC",
            model_lib:
                modelLibURLPrefix +
                modelVersion +
                "/Llama-3.2-3B-Instruct-q4f32_1-ctx4k_cs1k-webgpu.wasm",
            vram_required_MB: 2951.51,
            low_resource_required: true,
            overrides: {
                context_window_size: 4096,
            },
        },
        {
            model: "https://huggingface.co/mlc-ai/Hermes-3-Llama-3.2-3B-q4f16_1-MLC",
            model_id: "Hermes-3-Llama-3.2-3B-q4f16_1-MLC",
            model_lib:
                modelLibURLPrefix +
                modelVersion +
                "/Llama-3.2-3B-Instruct-q4f16_1-ctx4k_cs1k-webgpu.wasm",
            vram_required_MB: 2263.69,
            low_resource_required: true,
            overrides: {
                context_window_size: 4096,
            },
        },
        {
            model: "https://huggingface.co/mlc-ai/Hermes-3-Llama-3.1-8B-q4f32_1-MLC",
            model_id: "Hermes-3-Llama-3.1-8B-q4f32_1-MLC",
            model_lib:
                modelLibURLPrefix +
                modelVersion +
                "/Llama-3_1-8B-Instruct-q4f32_1-ctx4k_cs1k-webgpu.wasm",
            vram_required_MB: 5779.27,
            low_resource_required: false,
            overrides: {
                context_window_size: 4096,
            },
        },
        {
            model: "https://huggingface.co/mlc-ai/Hermes-3-Llama-3.1-8B-q4f16_1-MLC",
            model_id: "Hermes-3-Llama-3.1-8B-q4f16_1-MLC",
            model_lib:
                modelLibURLPrefix +
                modelVersion +
                "/Llama-3_1-8B-Instruct-q4f16_1-ctx4k_cs1k-webgpu.wasm",
            vram_required_MB: 4876.13,
            low_resource_required: false,
            overrides: {
                context_window_size: 4096,
            },
        },
        {
            model:
                "https://huggingface.co/mlc-ai/Hermes-2-Pro-Mistral-7B-q4f16_1-MLC",
            model_id: "Hermes-2-Pro-Mistral-7B-q4f16_1-MLC",
            model_lib:
                modelLibURLPrefix +
                modelVersion +
                "/Mistral-7B-Instruct-v0.3-q4f16_1-ctx4k_cs1k-webgpu.wasm",
            vram_required_MB: 4033.28,
            low_resource_required: false,
            required_features: ["shader-f16"],
            overrides: {
                context_window_size: 4096,
                sliding_window_size: -1,
            },
        },
        {
            model: "https://huggingface.co/mlc-ai/Phi-3.5-mini-instruct-q4f16_1-MLC",
            model_id: "Phi-3.5-mini-instruct-q4f16_1-MLC",
            model_lib:
                modelLibURLPrefix +
                modelVersion +
                "/Phi-3.5-mini-instruct-q4f16_1-ctx4k_cs1k-webgpu.wasm",
            vram_required_MB: 3672.07,
            low_resource_required: false,
            overrides: {
                context_window_size: 4096,
            },
        },
        {
            model: "https://huggingface.co/mlc-ai/Phi-3.5-mini-instruct-q4f32_1-MLC",
            model_id: "Phi-3.5-mini-instruct-q4f32_1-MLC",
            model_lib:
                modelLibURLPrefix +
                modelVersion +
                "/Phi-3.5-mini-instruct-q4f32_1-ctx4k_cs1k-webgpu.wasm",
            vram_required_MB: 5483.12,
            low_resource_required: false,
            overrides: {
                context_window_size: 4096,
            },
        },
        {
            model: "https://huggingface.co/mlc-ai/Phi-3.5-mini-instruct-q4f16_1-MLC",
            model_id: "Phi-3.5-mini-instruct-q4f16_1-MLC-1k",
            model_lib:
                modelLibURLPrefix +
                modelVersion +
                "/Phi-3.5-mini-instruct-q4f16_1-ctx4k_cs1k-webgpu.wasm",
            vram_required_MB: 2520.07,
            low_resource_required: true,
            overrides: {
                context_window_size: 1024,
            },
        },
        {
            model: "https://huggingface.co/mlc-ai/Phi-3.5-mini-instruct-q4f32_1-MLC",
            model_id: "Phi-3.5-mini-instruct-q4f32_1-MLC-1k",
            model_lib:
                modelLibURLPrefix +
                modelVersion +
                "/Phi-3.5-mini-instruct-q4f32_1-ctx4k_cs1k-webgpu.wasm",
            vram_required_MB: 3179.12,
            low_resource_required: true,
            overrides: {
                context_window_size: 1024,
            },
        },
        {
            model:
                "https://huggingface.co/mlc-ai/Mistral-7B-Instruct-v0.3-q4f16_1-MLC",
            model_id: "Mistral-7B-Instruct-v0.3-q4f16_1-MLC",
            model_lib:
                modelLibURLPrefix +
                modelVersion +
                "/Mistral-7B-Instruct-v0.3-q4f16_1-ctx4k_cs1k-webgpu.wasm",
            vram_required_MB: 4573.39,
            low_resource_required: false,
            required_features: ["shader-f16"],
            overrides: {
                context_window_size: 4096,
                sliding_window_size: -1,
            },
        },
        {
            model:
                "https://huggingface.co/mlc-ai/Mistral-7B-Instruct-v0.3-q4f32_1-MLC",
            model_id: "Mistral-7B-Instruct-v0.3-q4f32_1-MLC",
            model_lib:
                modelLibURLPrefix +
                modelVersion +
                "/Mistral-7B-Instruct-v0.3-q4f32_1-ctx4k_cs1k-webgpu.wasm",
            vram_required_MB: 5619.27,
            low_resource_required: false,
            overrides: {
                context_window_size: 4096,
                sliding_window_size: -1,
            },
        },
        {
            model:
                "https://huggingface.co/mlc-ai/Mistral-7B-Instruct-v0.2-q4f16_1-MLC",
            model_id: "Mistral-7B-Instruct-v0.2-q4f16_1-MLC",
            model_lib:
                modelLibURLPrefix +
                modelVersion +
                "/Mistral-7B-Instruct-v0.3-q4f16_1-ctx4k_cs1k-webgpu.wasm",
            vram_required_MB: 4573.39,
            low_resource_required: false,
            required_features: ["shader-f16"],
            overrides: {
                context_window_size: 4096,
                sliding_window_size: -1,
            },
        },
        {
            model:
                "https://huggingface.co/mlc-ai/OpenHermes-2.5-Mistral-7B-q4f16_1-MLC",
            model_id: "OpenHermes-2.5-Mistral-7B-q4f16_1-MLC",
            model_lib:
                modelLibURLPrefix +
                modelVersion +
                "/Mistral-7B-Instruct-v0.3-q4f16_1-ctx4k_cs1k-webgpu.wasm",
            vram_required_MB: 4573.39,
            low_resource_required: false,
            required_features: ["shader-f16"],
            overrides: {
                context_window_size: 4096,
                sliding_window_size: -1,
            },
        },
        {
            model:
                "https://huggingface.co/mlc-ai/NeuralHermes-2.5-Mistral-7B-q4f16_1-MLC",
            model_id: "NeuralHermes-2.5-Mistral-7B-q4f16_1-MLC",
            model_lib:
                modelLibURLPrefix +
                modelVersion +
                "/Mistral-7B-Instruct-v0.3-q4f16_1-ctx4k_cs1k-webgpu.wasm",
            vram_required_MB: 4573.39,
            low_resource_required: false,
            required_features: ["shader-f16"],
            overrides: {
                context_window_size: 4096,
                sliding_window_size: -1,
            },
        },
        {
            model: "https://huggingface.co/mlc-ai/WizardMath-7B-V1.1-q4f16_1-MLC",
            model_id: "WizardMath-7B-V1.1-q4f16_1-MLC",
            model_lib:
                modelLibURLPrefix +
                modelVersion +
                "/Mistral-7B-Instruct-v0.3-q4f16_1-ctx4k_cs1k-webgpu.wasm",
            vram_required_MB: 4573.39,
            low_resource_required: false,
            required_features: ["shader-f16"],
            overrides: {
                context_window_size: 4096,
                sliding_window_size: -1,
            },
        },
        {
            model: "https://huggingface.co/mlc-ai/SmolLM2-1.7B-Instruct-q4f16_1-MLC",
            model_id: "SmolLM2-1.7B-Instruct-q4f16_1-MLC",
            model_lib:
                modelLibURLPrefix +
                modelVersion +
                "/SmolLM2-1.7B-Instruct-q4f16_1-ctx4k_cs1k-webgpu.wasm",
            vram_required_MB: 1774.19,
            low_resource_required: true,
            required_features: ["shader-f16"],
            overrides: {
                context_window_size: 4096,
            },
        },
        {
            model: "https://huggingface.co/mlc-ai/SmolLM2-1.7B-Instruct-q4f32_1-MLC",
            model_id: "SmolLM2-1.7B-Instruct-q4f32_1-MLC",
            model_lib:
                modelLibURLPrefix +
                modelVersion +
                "/SmolLM2-1.7B-Instruct-q4f32_1-ctx4k_cs1k-webgpu.wasm",
            vram_required_MB: 2692.38,
            low_resource_required: true,
            overrides: {
                context_window_size: 4096,
            },
        },

        {
            model: "https://huggingface.co/mlc-ai/SmolLM2-360M-Instruct-q0f16-MLC",
            model_id: "SmolLM2-360M-Instruct-q0f16-MLC",
            model_lib:
                modelLibURLPrefix +
                modelVersion +
                "/SmolLM2-360M-Instruct-q0f16-ctx4k_cs1k-webgpu.wasm",
            vram_required_MB: 871.99,
            low_resource_required: true,
            required_features: ["shader-f16"],
            overrides: {
                context_window_size: 4096,
            },
        },
        {
            model: "https://huggingface.co/mlc-ai/SmolLM2-360M-Instruct-q0f32-MLC",
            model_id: "SmolLM2-360M-Instruct-q0f32-MLC",
            model_lib:
                modelLibURLPrefix +
                modelVersion +
                "/SmolLM2-360M-Instruct-q0f32-ctx4k_cs1k-webgpu.wasm",
            vram_required_MB: 1743.99,
            low_resource_required: true,
            overrides: {
                context_window_size: 4096,
            },
        },
        {
            model: "https://huggingface.co/mlc-ai/SmolLM2-360M-Instruct-q4f16_1-MLC",
            model_id: "SmolLM2-360M-Instruct-q4f16_1-MLC",
            model_lib:
                modelLibURLPrefix +
                modelVersion +
                "/SmolLM2-360M-Instruct-q4f16_1-ctx4k_cs1k-webgpu.wasm",
            vram_required_MB: 376.06,
            low_resource_required: true,
            required_features: ["shader-f16"],
            overrides: {
                context_window_size: 4096,
            },
        },
        {
            model: "https://huggingface.co/mlc-ai/SmolLM2-360M-Instruct-q4f32_1-MLC",
            model_id: "SmolLM2-360M-Instruct-q4f32_1-MLC",
            model_lib:
                modelLibURLPrefix +
                modelVersion +
                "/SmolLM2-360M-Instruct-q4f32_1-ctx4k_cs1k-webgpu.wasm",
            vram_required_MB: 579.61,
            low_resource_required: true,
            overrides: {
                context_window_size: 4096,
            },
        },
        {
            model: "https://huggingface.co/mlc-ai/SmolLM2-135M-Instruct-q0f16-MLC",
            model_id: "SmolLM2-135M-Instruct-q0f16-MLC",
            model_lib:
                modelLibURLPrefix +
                modelVersion +
                "/SmolLM2-135M-Instruct-q0f16-ctx4k_cs1k-webgpu.wasm",
            vram_required_MB: 359.69,
            low_resource_required: true,
            required_features: ["shader-f16"],
            overrides: {
                context_window_size: 4096,
            },
        },
        {
            model: "https://huggingface.co/mlc-ai/SmolLM2-135M-Instruct-q0f32-MLC",
            model_id: "SmolLM2-135M-Instruct-q0f32-MLC",
            model_lib:
                modelLibURLPrefix +
                modelVersion +
                "/SmolLM2-135M-Instruct-q0f32-ctx4k_cs1k-webgpu.wasm",
            vram_required_MB: 719.38,
            low_resource_required: true,
            overrides: {
                context_window_size: 4096,
            },
        },
        {
            model: "https://huggingface.co/mlc-ai/gemma-2-2b-it-q4f16_1-MLC",
            model_id: "gemma-2-2b-it-q4f16_1-MLC",
            model_lib:
                modelLibURLPrefix +
                modelVersion +
                "/gemma-2-2b-it-q4f16_1-ctx4k_cs1k-webgpu.wasm",
            vram_required_MB: 1895.3,
            low_resource_required: false,
            required_features: ["shader-f16"],
            overrides: {
                context_window_size: 4096,
            },
        },
        {
            model: "https://huggingface.co/mlc-ai/gemma-2-2b-it-q4f32_1-MLC",
            model_id: "gemma-2-2b-it-q4f32_1-MLC",
            model_lib:
                modelLibURLPrefix +
                modelVersion +
                "/gemma-2-2b-it-q4f32_1-ctx4k_cs1k-webgpu.wasm",
            vram_required_MB: 2508.75,
            low_resource_required: false,
            overrides: {
                context_window_size: 4096,
            },
        },
        {
            model: "https://huggingface.co/mlc-ai/gemma-2-2b-it-q4f16_1-MLC",
            model_id: "gemma-2-2b-it-q4f16_1-MLC-1k",
            model_lib:
                modelLibURLPrefix +
                modelVersion +
                "/gemma-2-2b-it-q4f16_1-ctx4k_cs1k-webgpu.wasm",
            vram_required_MB: 1583.3,
            low_resource_required: true,
            required_features: ["shader-f16"],
            overrides: {
                context_window_size: 1024,
            },
        },
        {
            model: "https://huggingface.co/mlc-ai/gemma-2-2b-it-q4f32_1-MLC",
            model_id: "gemma-2-2b-it-q4f32_1-MLC-1k",
            model_lib:
                modelLibURLPrefix +
                modelVersion +
                "/gemma-2-2b-it-q4f32_1-ctx4k_cs1k-webgpu.wasm",
            vram_required_MB: 1884.75,
            low_resource_required: true,
            overrides: {
                context_window_size: 1024,
            },
        },
        {
            model: "https://huggingface.co/mlc-ai/gemma-2-9b-it-q4f16_1-MLC",
            model_id: "gemma-2-9b-it-q4f16_1-MLC",
            model_lib:
                modelLibURLPrefix +
                modelVersion +
                "/gemma-2-9b-it-q4f16_1-ctx4k_cs1k-webgpu.wasm",
            vram_required_MB: 6422.01,
            low_resource_required: false,
            required_features: ["shader-f16"],
            overrides: {
                context_window_size: 4096,
            },
        },
        {
            model: "https://huggingface.co/mlc-ai/gemma-2-9b-it-q4f32_1-MLC",
            model_id: "gemma-2-9b-it-q4f32_1-MLC",
            model_lib:
                modelLibURLPrefix +
                modelVersion +
                "/gemma-2-9b-it-q4f32_1-ctx4k_cs1k-webgpu.wasm",
            vram_required_MB: 8383.33,
            low_resource_required: false,
            overrides: {
                context_window_size: 4096,
            },
        },
        {
            model: "https://huggingface.co/mlc-ai/gemma-2-2b-jpn-it-q4f16_1-MLC",
            model_id: "gemma-2-2b-jpn-it-q4f16_1-MLC",
            model_lib:
                modelLibURLPrefix +
                modelVersion +
                "/gemma-2-2b-jpn-it-q4f16_1-ctx4k_cs1k-webgpu.wasm",
            vram_required_MB: 1895.3,
            low_resource_required: true,
            required_features: ["shader-f16"],
            overrides: {
                context_window_size: 4096,
            },
        },
        {
            model: "https://huggingface.co/mlc-ai/gemma-2-2b-jpn-it-q4f32_1-MLC",
            model_id: "gemma-2-2b-jpn-it-q4f32_1-MLC",
            model_lib:
                modelLibURLPrefix +
                modelVersion +
                "/gemma-2-2b-jpn-it-q4f32_1-ctx4k_cs1k-webgpu.wasm",
            vram_required_MB: 2508.75,
            low_resource_required: true,
            overrides: {
                context_window_size: 4096,
            },
        },
        {
            model: "https://huggingface.co/mlc-ai/Qwen3-0.6B-q4f16_1-MLC",
            model_id: "Qwen3-0.6B-q4f16_1-MLC",
            model_lib:
                modelLibURLPrefix +
                modelVersion +
                "/Qwen3-0.6B-q4f16_1-ctx4k_cs1k-webgpu.wasm",
            vram_required_MB: 1403.34,
            low_resource_required: true,
            overrides: {
                context_window_size: 4096,
            },
        },
        {
            model: "https://huggingface.co/mlc-ai/Qwen3-0.6B-q4f32_1-MLC",
            model_id: "Qwen3-0.6B-q4f32_1-MLC",
            model_lib:
                modelLibURLPrefix +
                modelVersion +
                "/Qwen3-0.6B-q4f32_1-ctx4k_cs1k-webgpu.wasm",
            vram_required_MB: 1924.98,
            low_resource_required: true,
            overrides: {
                context_window_size: 4096,
            },
        },
        {
            model: "https://huggingface.co/mlc-ai/Qwen3-0.6B-q0f16-MLC",
            model_id: "Qwen3-0.6B-q0f16-MLC",
            model_lib:
                modelLibURLPrefix +
                modelVersion +
                "/Qwen3-0.6B-q0f16-ctx4k_cs1k-webgpu.wasm",
            vram_required_MB: 2220.38,
            low_resource_required: true,
            overrides: {
                context_window_size: 4096,
            },
        },
        {
            model: "https://huggingface.co/mlc-ai/Qwen3-0.6B-q0f32-MLC",
            model_id: "Qwen3-0.6B-q0f32-MLC",
            model_lib:
                modelLibURLPrefix +
                modelVersion +
                "/Qwen3-0.6B-q0f32-ctx4k_cs1k-webgpu.wasm",
            vram_required_MB: 3843.25,
            low_resource_required: true,
            overrides: {
                context_window_size: 4096,
            },
        },
        {
            model: "https://huggingface.co/mlc-ai/Qwen3-1.7B-q4f16_1-MLC",
            model_id: "Qwen3-1.7B-q4f16_1-MLC",
            model_lib:
                modelLibURLPrefix +
                modelVersion +
                "/Qwen3-1.7B-q4f16_1-ctx4k_cs1k-webgpu.wasm",
            vram_required_MB: 2036.66,
            low_resource_required: true,
            overrides: {
                context_window_size: 4096,
            },
        },
        {
            model: "https://huggingface.co/mlc-ai/Qwen3-1.7B-q4f32_1-MLC",
            model_id: "Qwen3-1.7B-q4f32_1-MLC",
            model_lib:
                modelLibURLPrefix +
                modelVersion +
                "/Qwen3-1.7B-q4f32_1-ctx4k_cs1k-webgpu.wasm",
            vram_required_MB: 2635.44,
            low_resource_required: true,
            overrides: {
                context_window_size: 4096,
            },
        },
        {
            model: "https://huggingface.co/mlc-ai/Qwen3-4B-q4f16_1-MLC",
            model_id: "Qwen3-4B-q4f16_1-MLC",
            model_lib:
                modelLibURLPrefix +
                modelVersion +
                "/Qwen3-4B-q4f16_1-ctx4k_cs1k-webgpu.wasm",
            vram_required_MB: 3431.59,
            low_resource_required: true,
            overrides: {
                context_window_size: 4096,
            },
        },
        {
            model: "https://huggingface.co/mlc-ai/Qwen3-4B-q4f32_1-MLC",
            model_id: "Qwen3-4B-q4f32_1-MLC",
            model_lib:
                modelLibURLPrefix +
                modelVersion +
                "/Qwen3-4B-q4f32_1-ctx4k_cs1k-webgpu.wasm",
            vram_required_MB: 4327.71,
            low_resource_required: true,
            overrides: {
                context_window_size: 4096,
            },
        },
        {
            model: "https://huggingface.co/mlc-ai/Qwen3-8B-q4f16_1-MLC",
            model_id: "Qwen3-8B-q4f16_1-MLC",
            model_lib:
                modelLibURLPrefix +
                modelVersion +
                "/Qwen3-8B-q4f16_1-ctx4k_cs1k-webgpu.wasm",
            vram_required_MB: 5695.78,
            low_resource_required: false,
            overrides: {
                context_window_size: 4096,
            },
        },
        {
            model: "https://huggingface.co/mlc-ai/Qwen3-8B-q4f32_1-MLC",
            model_id: "Qwen3-8B-q4f32_1-MLC",
            model_lib:
                modelLibURLPrefix +
                modelVersion +
                "/Qwen3-8B-q4f32_1-ctx4k_cs1k-webgpu.wasm",
            vram_required_MB: 6852.55,
            low_resource_required: false,
            overrides: {
                context_window_size: 4096,
            },
        },
        {
            model: "https://huggingface.co/mlc-ai/Qwen2.5-0.5B-Instruct-q4f16_1-MLC",
            model_id: "Qwen2.5-0.5B-Instruct-q4f16_1-MLC",
            model_lib:
                modelLibURLPrefix +
                modelVersion +
                "/Qwen2-0.5B-Instruct-q4f16_1-ctx4k_cs1k-webgpu.wasm",
            low_resource_required: true,
            vram_required_MB: 944.62,
            overrides: {
                context_window_size: 4096,
            },
        },
        {
            model: "https://huggingface.co/mlc-ai/Qwen2.5-0.5B-Instruct-q4f32_1-MLC",
            model_id: "Qwen2.5-0.5B-Instruct-q4f32_1-MLC",
            model_lib:
                modelLibURLPrefix +
                modelVersion +
                "/Qwen2-0.5B-Instruct-q4f32_1-ctx4k_cs1k-webgpu.wasm",
            low_resource_required: true,
            vram_required_MB: 1060.2,
            overrides: {
                context_window_size: 4096,
            },
        },
        {
            model: "https://huggingface.co/mlc-ai/Qwen2.5-0.5B-Instruct-q0f16-MLC",
            model_id: "Qwen2.5-0.5B-Instruct-q0f16-MLC",
            model_lib:
                modelLibURLPrefix +
                modelVersion +
                "/Qwen2-0.5B-Instruct-q0f16-ctx4k_cs1k-webgpu.wasm",
            low_resource_required: true,
            vram_required_MB: 1624.12,
            overrides: {
                context_window_size: 4096,
            },
        },
        {
            model: "https://huggingface.co/mlc-ai/Qwen2.5-0.5B-Instruct-q0f32-MLC",
            model_id: "Qwen2.5-0.5B-Instruct-q0f32-MLC",
            model_lib:
                modelLibURLPrefix +
                modelVersion +
                "/Qwen2-0.5B-Instruct-q0f32-ctx4k_cs1k-webgpu.wasm",
            low_resource_required: true,
            vram_required_MB: 2654.75,
            overrides: {
                context_window_size: 4096,
            },
        },
        {
            model: "https://huggingface.co/mlc-ai/Qwen2.5-1.5B-Instruct-q4f16_1-MLC",
            model_id: "Qwen2.5-1.5B-Instruct-q4f16_1-MLC",
            model_lib:
                modelLibURLPrefix +
                modelVersion +
                "/Qwen2-1.5B-Instruct-q4f16_1-ctx4k_cs1k-webgpu.wasm",
            low_resource_required: true,
            vram_required_MB: 1629.75,
            overrides: {
                context_window_size: 4096,
            },
        },
        {
            model: "https://huggingface.co/mlc-ai/Qwen2.5-1.5B-Instruct-q4f32_1-MLC",
            model_id: "Qwen2.5-1.5B-Instruct-q4f32_1-MLC",
            model_lib:
                modelLibURLPrefix +
                modelVersion +
                "/Qwen2-1.5B-Instruct-q4f32_1-ctx4k_cs1k-webgpu.wasm",
            low_resource_required: true,
            vram_required_MB: 1888.97,
            overrides: {
                context_window_size: 4096,
            },
        },
        {
            model: "https://huggingface.co/mlc-ai/Qwen2.5-3B-Instruct-q4f16_1-MLC",
            model_id: "Qwen2.5-3B-Instruct-q4f16_1-MLC",
            model_lib:
                modelLibURLPrefix +
                modelVersion +
                "/Qwen2.5-3B-Instruct-q4f16_1-ctx4k_cs1k-webgpu.wasm",
            low_resource_required: true,
            vram_required_MB: 2504.76,
            overrides: {
                context_window_size: 4096,
            },
        },
        {
            model: "https://huggingface.co/mlc-ai/Qwen2.5-3B-Instruct-q4f32_1-MLC",
            model_id: "Qwen2.5-3B-Instruct-q4f32_1-MLC",
            model_lib:
                modelLibURLPrefix +
                modelVersion +
                "/Qwen2.5-3B-Instruct-q4f32_1-ctx4k_cs1k-webgpu.wasm",
            low_resource_required: true,
            vram_required_MB: 2893.64,
            overrides: {
                context_window_size: 4096,
            },
        },
        {
            model: "https://huggingface.co/mlc-ai/Qwen2.5-7B-Instruct-q4f16_1-MLC",
            model_id: "Qwen2.5-7B-Instruct-q4f16_1-MLC",
            model_lib:
                modelLibURLPrefix +
                modelVersion +
                "/Qwen2-7B-Instruct-q4f16_1-ctx4k_cs1k-webgpu.wasm",
            low_resource_required: false,
            vram_required_MB: 5106.67,
            overrides: {
                context_window_size: 4096,
            },
        },
        {
            model: "https://huggingface.co/mlc-ai/Qwen2.5-7B-Instruct-q4f32_1-MLC",
            model_id: "Qwen2.5-7B-Instruct-q4f32_1-MLC",
            model_lib:
                modelLibURLPrefix +
                modelVersion +
                "/Qwen2-7B-Instruct-q4f32_1-ctx4k_cs1k-webgpu.wasm",
            low_resource_required: false,
            vram_required_MB: 5900.09,
            overrides: {
                context_window_size: 4096,
            },
        },
        {
            model:
                "https://huggingface.co/mlc-ai/Qwen2.5-Coder-0.5B-Instruct-q4f16_1-MLC",
            model_id: "Qwen2.5-Coder-0.5B-Instruct-q4f16_1-MLC",
            model_lib:
                modelLibURLPrefix +
                modelVersion +
                "/Qwen2-0.5B-Instruct-q4f16_1-ctx4k_cs1k-webgpu.wasm",
            low_resource_required: true,
            vram_required_MB: 944.62,
            overrides: {
                context_window_size: 4096,
            },
        },
        {
            model:
                "https://huggingface.co/mlc-ai/Qwen2.5-Coder-0.5B-Instruct-q4f32_1-MLC",
            model_id: "Qwen2.5-Coder-0.5B-Instruct-q4f32_1-MLC",
            model_lib:
                modelLibURLPrefix +
                modelVersion +
                "/Qwen2-0.5B-Instruct-q4f32_1-ctx4k_cs1k-webgpu.wasm",
            low_resource_required: true,
            vram_required_MB: 1060.2,
            overrides: {
                context_window_size: 4096,
            },
        },
        {
            model:
                "https://huggingface.co/mlc-ai/Qwen2.5-Coder-0.5B-Instruct-q0f16-MLC",
            model_id: "Qwen2.5-Coder-0.5B-Instruct-q0f16-MLC",
            model_lib:
                modelLibURLPrefix +
                modelVersion +
                "/Qwen2-0.5B-Instruct-q0f16-ctx4k_cs1k-webgpu.wasm",
            low_resource_required: true,
            vram_required_MB: 1624.12,
            overrides: {
                context_window_size: 4096,
            },
        },
        {
            model:
                "https://huggingface.co/mlc-ai/Qwen2.5-Coder-0.5B-Instruct-q0f32-MLC",
            model_id: "Qwen2.5-Coder-0.5B-Instruct-q0f32-MLC",
            model_lib:
                modelLibURLPrefix +
                modelVersion +
                "/Qwen2-0.5B-Instruct-q0f32-ctx4k_cs1k-webgpu.wasm",
            low_resource_required: true,
            vram_required_MB: 2654.75,
            overrides: {
                context_window_size: 4096,
            },
        },
        {
            model:
                "https://huggingface.co/mlc-ai/Qwen2.5-Coder-1.5B-Instruct-q4f16_1-MLC",
            model_id: "Qwen2.5-Coder-1.5B-Instruct-q4f16_1-MLC",
            model_lib:
                modelLibURLPrefix +
                modelVersion +
                "/Qwen2-1.5B-Instruct-q4f16_1-ctx4k_cs1k-webgpu.wasm",
            low_resource_required: false,
            vram_required_MB: 1629.75,
            overrides: {
                context_window_size: 4096,
            },
        },
        {
            model:
                "https://huggingface.co/mlc-ai/Qwen2.5-Coder-1.5B-Instruct-q4f32_1-MLC",
            model_id: "Qwen2.5-Coder-1.5B-Instruct-q4f32_1-MLC",
            model_lib:
                modelLibURLPrefix +
                modelVersion +
                "/Qwen2-1.5B-Instruct-q4f32_1-ctx4k_cs1k-webgpu.wasm",
            low_resource_required: false,
            vram_required_MB: 1888.97,
            overrides: {
                context_window_size: 4096,
            },
        },
        {
            model:
                "https://huggingface.co/mlc-ai/Qwen2.5-Coder-3B-Instruct-q4f16_1-MLC",
            model_id: "Qwen2.5-Coder-3B-Instruct-q4f16_1-MLC",
            model_lib:
                modelLibURLPrefix +
                modelVersion +
                "/Qwen2.5-3B-Instruct-q4f16_1-ctx4k_cs1k-webgpu.wasm",
            low_resource_required: true,
            vram_required_MB: 2504.76,
            overrides: {
                context_window_size: 4096,
            },
        },
        {
            model:
                "https://huggingface.co/mlc-ai/Qwen2.5-Coder-3B-Instruct-q4f32_1-MLC",
            model_id: "Qwen2.5-Coder-3B-Instruct-q4f32_1-MLC",
            model_lib:
                modelLibURLPrefix +
                modelVersion +
                "/Qwen2.5-3B-Instruct-q4f32_1-ctx4k_cs1k-webgpu.wasm",
            low_resource_required: true,
            vram_required_MB: 2893.64,
            overrides: {
                context_window_size: 4096,
            },
        },
        {
            model:
                "https://huggingface.co/mlc-ai/Qwen2.5-Coder-7B-Instruct-q4f16_1-MLC",
            model_id: "Qwen2.5-Coder-7B-Instruct-q4f16_1-MLC",
            model_lib:
                modelLibURLPrefix +
                modelVersion +
                "/Qwen2-7B-Instruct-q4f16_1-ctx4k_cs1k-webgpu.wasm",
            low_resource_required: false,
            vram_required_MB: 5106.67,
            overrides: {
                context_window_size: 4096,
            },
        },
        {
            model:
                "https://huggingface.co/mlc-ai/Qwen2.5-Coder-7B-Instruct-q4f32_1-MLC",
            model_id: "Qwen2.5-Coder-7B-Instruct-q4f32_1-MLC",
            model_lib:
                modelLibURLPrefix +
                modelVersion +
                "/Qwen2-7B-Instruct-q4f32_1-ctx4k_cs1k-webgpu.wasm",
            low_resource_required: false,
            vram_required_MB: 5900.09,
            overrides: {
                context_window_size: 4096,
            },
        },
        {
            model:
                "https://huggingface.co/mlc-ai/Qwen2.5-Math-1.5B-Instruct-q4f16_1-MLC",
            model_id: "Qwen2.5-Math-1.5B-Instruct-q4f16_1-MLC",
            model_lib:
                modelLibURLPrefix +
                modelVersion +
                "/Qwen2-1.5B-Instruct-q4f16_1-ctx4k_cs1k-webgpu.wasm",
            low_resource_required: true,
            vram_required_MB: 1629.75,
            overrides: {
                context_window_size: 4096,
            },
        },
        {
            model:
                "https://huggingface.co/mlc-ai/Qwen2.5-Math-1.5B-Instruct-q4f32_1-MLC",
            model_id: "Qwen2.5-Math-1.5B-Instruct-q4f32_1-MLC",
            model_lib:
                modelLibURLPrefix +
                modelVersion +
                "/Qwen2-1.5B-Instruct-q4f32_1-ctx4k_cs1k-webgpu.wasm",
            low_resource_required: true,
            vram_required_MB: 1888.97,
            overrides: {
                context_window_size: 4096,
            },
        },
        {
            model: "https://huggingface.co/mlc-ai/stablelm-2-zephyr-1_6b-q4f16_1-MLC",
            model_id: "stablelm-2-zephyr-1_6b-q4f16_1-MLC",
            model_lib:
                modelLibURLPrefix +
                modelVersion +
                "/stablelm-2-zephyr-1_6b-q4f16_1-ctx4k_cs1k-webgpu.wasm",
            vram_required_MB: 2087.66,
            low_resource_required: false,
            overrides: {
                context_window_size: 4096,
            },
        },
        {
            model: "https://huggingface.co/mlc-ai/stablelm-2-zephyr-1_6b-q4f32_1-MLC",
            model_id: "stablelm-2-zephyr-1_6b-q4f32_1-MLC",
            model_lib:
                modelLibURLPrefix +
                modelVersion +
                "/stablelm-2-zephyr-1_6b-q4f32_1-ctx4k_cs1k-webgpu.wasm",
            vram_required_MB: 2999.33,
            low_resource_required: false,
            overrides: {
                context_window_size: 4096,
            },
        },
        {
            model: "https://huggingface.co/mlc-ai/stablelm-2-zephyr-1_6b-q4f16_1-MLC",
            model_id: "stablelm-2-zephyr-1_6b-q4f16_1-MLC-1k",
            model_lib:
                modelLibURLPrefix +
                modelVersion +
                "/stablelm-2-zephyr-1_6b-q4f16_1-ctx4k_cs1k-webgpu.wasm",
            vram_required_MB: 1511.66,
            low_resource_required: true,
            overrides: {
                context_window_size: 1024,
            },
        },
        {
            model: "https://huggingface.co/mlc-ai/stablelm-2-zephyr-1_6b-q4f32_1-MLC",
            model_id: "stablelm-2-zephyr-1_6b-q4f32_1-MLC-1k",
            model_lib:
                modelLibURLPrefix +
                modelVersion +
                "/stablelm-2-zephyr-1_6b-q4f32_1-ctx4k_cs1k-webgpu.wasm",
            vram_required_MB: 1847.33,
            low_resource_required: true,
            overrides: {
                context_window_size: 1024,
            },
        },
        {
            model:
                "https://huggingface.co/mlc-ai/RedPajama-INCITE-Chat-3B-v1-q4f16_1-MLC",
            model_id: "RedPajama-INCITE-Chat-3B-v1-q4f16_1-MLC",
            model_lib:
                modelLibURLPrefix +
                modelVersion +
                "/RedPajama-INCITE-Chat-3B-v1-q4f16_1-ctx2k_cs1k-webgpu.wasm",
            vram_required_MB: 2972.09,
            low_resource_required: false,
            required_features: ["shader-f16"],
            overrides: {
                context_window_size: 2048,
            },
        },
        {
            model:
                "https://huggingface.co/mlc-ai/RedPajama-INCITE-Chat-3B-v1-q4f32_1-MLC",
            model_id: "RedPajama-INCITE-Chat-3B-v1-q4f32_1-MLC",
            model_lib:
                modelLibURLPrefix +
                modelVersion +
                "/RedPajama-INCITE-Chat-3B-v1-q4f32_1-ctx2k_cs1k-webgpu.wasm",
            vram_required_MB: 3928.09,
            low_resource_required: false,
            overrides: {
                context_window_size: 2048,
            },
        },
        {
            model:
                "https://huggingface.co/mlc-ai/RedPajama-INCITE-Chat-3B-v1-q4f16_1-MLC",
            model_id: "RedPajama-INCITE-Chat-3B-v1-q4f16_1-MLC-1k",
            model_lib:
                modelLibURLPrefix +
                modelVersion +
                "/RedPajama-INCITE-Chat-3B-v1-q4f16_1-ctx2k_cs1k-webgpu.wasm",
            vram_required_MB: 2041.09,
            low_resource_required: true,
            required_features: ["shader-f16"],
            overrides: {
                context_window_size: 1024,
            },
        },
        {
            model:
                "https://huggingface.co/mlc-ai/RedPajama-INCITE-Chat-3B-v1-q4f32_1-MLC",
            model_id: "RedPajama-INCITE-Chat-3B-v1-q4f32_1-MLC-1k",
            model_lib:
                modelLibURLPrefix +
                modelVersion +
                "/RedPajama-INCITE-Chat-3B-v1-q4f32_1-ctx2k_cs1k-webgpu.wasm",
            vram_required_MB: 2558.09,
            low_resource_required: true,
            overrides: {
                context_window_size: 1024,
            },
        },
        {
            model:
                "https://huggingface.co/mlc-ai/TinyLlama-1.1B-Chat-v1.0-q4f16_1-MLC",
            model_id: "TinyLlama-1.1B-Chat-v1.0-q4f16_1-MLC",
            model_lib:
                modelLibURLPrefix +
                modelVersion +
                "/TinyLlama-1.1B-Chat-v1.0-q4f16_1-ctx2k_cs1k-webgpu.wasm",
            vram_required_MB: 697.24,
            low_resource_required: true,
            required_features: ["shader-f16"],
            overrides: {
                context_window_size: 2048,
            },
        },
        {
            model:
                "https://huggingface.co/mlc-ai/TinyLlama-1.1B-Chat-v1.0-q4f32_1-MLC",
            model_id: "TinyLlama-1.1B-Chat-v1.0-q4f32_1-MLC",
            model_lib:
                modelLibURLPrefix +
                modelVersion +
                "/TinyLlama-1.1B-Chat-v1.0-q4f32_1-ctx2k_cs1k-webgpu.wasm",
            vram_required_MB: 839.98,
            low_resource_required: true,
            overrides: {
                context_window_size: 2048,
            },
        },
        {
            model:
                "https://huggingface.co/mlc-ai/TinyLlama-1.1B-Chat-v1.0-q4f16_1-MLC",
            model_id: "TinyLlama-1.1B-Chat-v1.0-q4f16_1-MLC-1k",
            model_lib:
                modelLibURLPrefix +
                modelVersion +
                "/TinyLlama-1.1B-Chat-v1.0-q4f16_1-ctx2k_cs1k-webgpu.wasm",
            vram_required_MB: 675.24,
            low_resource_required: true,
            required_features: ["shader-f16"],
            overrides: {
                context_window_size: 1024,
            },
        },
        {
            model:
                "https://huggingface.co/mlc-ai/TinyLlama-1.1B-Chat-v1.0-q4f32_1-MLC",
            model_id: "TinyLlama-1.1B-Chat-v1.0-q4f32_1-MLC-1k",
            model_lib:
                modelLibURLPrefix +
                modelVersion +
                "/TinyLlama-1.1B-Chat-v1.0-q4f32_1-ctx2k_cs1k-webgpu.wasm",
            vram_required_MB: 795.98,
            low_resource_required: true,
            overrides: {
                context_window_size: 1024,
            },
        },
    ]
};

function getSelectedModel() {
    const modelSelect = document.getElementById("model-select");
    return modelSelect ? modelSelect.value : "Llama-3.2-3B-Instruct-q4f16_1-MLC";
}

function updateEngineInitProgressCallback(report) {
    console.log("initialize", report.progress);
    document.getElementById("download-status").textContent = report.text;
}

let engine = null;

async function initializeWebLLMEngine() {
    const selectedModel = getSelectedModel();
    alert(`Begin to download the model: ${selectedModel}, this may take a while.`);

    document.getElementById("download-status").classList.remove("hidden");
    const config = {
        temperature: 1.0,
        top_p: 1,
    };

    try {
        if (engine) {
            engine = null;
        }

        engine = await webllm.CreateMLCEngine(
            selectedModel,
            {
                appConfig: myAppConfig,
                initProgressCallback: updateEngineInitProgressCallback
            }
        );

        alert(`Successfully downloaded the model: ${selectedModel}! You can now start chatting.`);
    } catch (error) {
        alert("Failed to download the model. Please check the console for details. " + error.message);
        console.error("ERROR: ", error);
        throw error;
    }
}

async function streamingGenerating(messages, onUpdate, onFinish, onError) {
    if (!engine) {
        onError(new Error("Engine not initialized. Please download a model first."));
        return;
    }

    try {
        let curMessage = "";
        const completion = await engine.chat.completions.create({
            stream: true,
            messages,
        });
        for await (const chunk of completion) {
            const curDelta = chunk.choices[0].delta.content;
            if (curDelta) {
                curMessage += curDelta;
            }
            onUpdate(curMessage);
        }
        const finalMessage = await engine.getMessage();
        onFinish(finalMessage);
    } catch (err) {
        onError(err);
    }
}

function onMessageSend() {
    const input = document.getElementById("user-input").value.trim();
    const message = {
        content: input,
        role: "user",
    };
    if (input.length === 0) {
        return;
    }
    document.getElementById("send").disabled = true;

    messages.push(message);
    appendMessage(message);

    document.getElementById("user-input").value = "";
    document
        .getElementById("user-input")
        .setAttribute("placeholder", "Generating...");

    const aiMessage = {
        content: "typing...",
        role: "assistant",
    };
    appendMessage(aiMessage);

    const onFinishGenerating = (finalMessage) => {
        updateLastMessage(finalMessage);
        document.getElementById("send").disabled = false;
        document
            .getElementById("user-input")
            .setAttribute("placeholder", "Type a message...");
        if (engine && engine.runtimeStatsText) {
            engine.runtimeStatsText().then(statsText => {
                document.getElementById('chat-stats').classList.remove('hidden');
                document.getElementById('chat-stats').textContent = statsText;
            });
        }
    };

    streamingGenerating(
        messages,
        updateLastMessage,
        onFinishGenerating,
        console.error,
    );
}

function appendMessage(message) {
    const chatBox = document.getElementById("chat-box");
    const container = document.createElement("div");
    container.classList.add("message-container");
    const newMessage = document.createElement("div");
    newMessage.classList.add("message");
    newMessage.textContent = message.content;

    if (message.role === "user") {
        container.classList.add("user");
    } else {
        container.classList.add("assistant");
    }

    container.appendChild(newMessage);
    chatBox.appendChild(container);
    chatBox.scrollTop = chatBox.scrollHeight;
}

function updateLastMessage(content) {
    const messageDoms = document
        .getElementById("chat-box")
        .querySelectorAll(".message");
    const lastMessageDom = messageDoms[messageDoms.length - 1];
    lastMessageDom.textContent = content;
}

async function getGPUInfo() {
    const gpuContent = document.getElementById("gpu-content");
    const now = new Date().toLocaleTimeString();

    try {
        let info = `[${now}] GPU Information (Browser Available)\n`;
        info += "================================================\n";

        if ('gpu' in navigator) {
            info += "WebGPU: Supported ✓\n";
            try {
                const adapter = await navigator.gpu.requestAdapter();
                if (adapter) {
                    const device = await adapter.requestDevice();
                    info += `Adapter: Available\n`;

                    const features = Array.from(adapter.features || []);
                    if (features.length > 0) {
                        info += `Features: ${features.join(', ')}\n`;
                    }

                    const limits = adapter.limits;
                    if (limits) {
                        info += `Max Texture Size: ${limits.maxTextureDimension2D || 'Unknown'}\n`;
                        info += `Max Buffer Size: ${limits.maxBufferSize ? (limits.maxBufferSize / 1024 / 1024).toFixed(0) + ' MB' : 'Unknown'}\n`;
                    }
                } else {
                    info += "Adapter: Not available\n";
                }
            } catch (e) {
                info += `WebGPU Error: ${e.message}\n`;
            }
        } else {
            info += "WebGPU: Not supported ✗\n";
        }

        const canvas = document.createElement('canvas');
        const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
        if (gl) {
            info += "\nWebGL Information:\n";
            const renderer = gl.getParameter(gl.RENDERER);
            const vendor = gl.getParameter(gl.VENDOR);
            const version = gl.getParameter(gl.VERSION);

            info += `Renderer: ${renderer}\n`;
            info += `Vendor: ${vendor}\n`;
            info += `Version: ${version}\n`;

            const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
            if (debugInfo) {
                const unmaskedRenderer = gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL);
                const unmaskedVendor = gl.getParameter(debugInfo.UNMASKED_VENDOR_WEBGL);
                info += `Unmasked Renderer: ${unmaskedRenderer}\n`;
                info += `Unmasked Vendor: ${unmaskedVendor}\n`;
            }
        } else {
            info += "WebGL: Not supported ✗\n";
        }

        if ('memory' in performance) {
            info += "\nMemory Usage:\n";
            const memory = performance.memory;
            info += `Used: ${(memory.usedJSHeapSize / 1024 / 1024).toFixed(2)} MB\n`;
            info += `Total: ${(memory.totalJSHeapSize / 1024 / 1024).toFixed(2)} MB\n`;
            info += `Limit: ${(memory.jsHeapSizeLimit / 1024 / 1024).toFixed(2)} MB\n`;
        }

        if ('storage' in navigator) {
            const estimate = await navigator.storage.estimate();
            info += "\nStorage Usage:\n";
            info += `Used: ${(estimate.usage / 1024 / 1024).toFixed(2)} MB\n`;
            info += `Quota: ${(estimate.quota / 1024 / 1024 / 1024).toFixed(2)} GB\n`;
        }

        info += "\nSystem Information:\n";
        info += `Platform: ${navigator.platform}\n`;
        info += `User Agent: ${navigator.userAgent.substring(0, 50)}...\n`;
        info += `Hardware Concurrency: ${navigator.hardwareConcurrency} cores\n`;

        info += "\nThis information is provided by the browser and may not be fully accurate.\n";

        gpuContent.textContent = info;
    } catch (error) {
        gpuContent.textContent = `Error getting GPU info: ${error.message}`;
    }
}

setInterval(getGPUInfo, 500);

window.addEventListener('DOMContentLoaded', getGPUInfo);

/*************** UI binding ***************/
document.addEventListener('DOMContentLoaded', function () {
    // 绑定下载按钮事件
    const downloadBtn = document.getElementById("download");
    if (downloadBtn) {
        downloadBtn.addEventListener("click", function () {
            console.log("Download button clicked"); // 添加调试日志
            downloadBtn.disabled = true;
            downloadBtn.textContent = "Downloading...";

            initializeWebLLMEngine().then(() => {
                document.getElementById("send").disabled = false;
                downloadBtn.disabled = false;
                downloadBtn.textContent = "Download Again";
            }).catch((error) => {
                downloadBtn.disabled = false;
                downloadBtn.textContent = "Download";
                console.error("ERROR: ", error);
            });
        });
    } else {
        console.error("Download button not found!");
    }

    // 绑定模型选择事件
    const modelSelect = document.getElementById("model-select");
    if (modelSelect) {
        modelSelect.addEventListener("change", function () {
            const selectedModel = getSelectedModel();
            console.log("Selected model:", selectedModel);

            if (engine) {
                document.getElementById("download-status").textContent = "Model changed. Please download the new model.";
                document.getElementById("send").disabled = true;
            }
        });
    } else {
        console.error("Model select not found!");
    }

    // 绑定发送按钮事件
    const sendBtn = document.getElementById("send");
    if (sendBtn) {
        sendBtn.addEventListener("click", function () {
            onMessageSend();
        });
    }

    // 绑定输入框回车事件
    const userInput = document.getElementById("user-input");
    if (userInput) {
        userInput.addEventListener("keypress", function (event) {
            if (event.key === "Enter" && !document.getElementById("send").disabled) {
                onMessageSend();
            }
        });
    }
});
