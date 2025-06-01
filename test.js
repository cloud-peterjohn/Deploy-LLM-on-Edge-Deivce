import * as webllm from "https://esm.run/@mlc-ai/web-llm";

/*************** WebLLM logic ***************/
const messages = [
    {
        content: "You are a helpful AI agent helping users.",
        role: "system",
    },
];

const selectedModel = "Llama-3.2-3B-Instruct-q4f16_1-MLC";

function updateEngineInitProgressCallback(report) {
    console.log("initialize", report.progress);
    document.getElementById("download-status").textContent = report.text;
}

const engine = new webllm.MLCEngine();
engine.setInitProgressCallback(updateEngineInitProgressCallback);

async function initializeWebLLMEngine() {
    alert("Begin to download the model, this may take a while.");

    document.getElementById("download-status").classList.remove("hidden");
    const config = {
        temperature: 1.0,
        top_p: 1,
    };

    try {
        await engine.reload(selectedModel, config);
        alert("Successfully downloaded the model! You can now start chatting.");
    } catch (error) {
        alert("Failed to download the model. Please check the console for details. " + error.message);
        console.error("ERROR: ", error);
        throw error;
    }
}

async function streamingGenerating(messages, onUpdate, onFinish, onError) {
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
        engine.runtimeStatsText().then(statsText => {
            document.getElementById('chat-stats').classList.remove('hidden');
            document.getElementById('chat-stats').textContent = statsText;
        })
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
document.getElementById("download").addEventListener("click", function () {
    const downloadBtn = document.getElementById("download");
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

document.getElementById("send").addEventListener("click", function () {
    onMessageSend();
});

document.getElementById("user-input").addEventListener("keypress", function (event) {
    if (event.key === "Enter" && !document.getElementById("send").disabled) {
        onMessageSend();
    }
});