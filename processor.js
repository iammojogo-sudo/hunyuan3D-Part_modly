"use strict";
const path = require("path");
const { spawn } = require("child_process");
const fs = require("fs");
const os = require("os");

const processor = async (input, params, context) => {
    const extDir = __dirname;
    const isWin  = process.platform === "win32";
    const pythonExe = isWin
        ? path.join(extDir, "venv", "Scripts", "python.exe")
        : path.join(extDir, "venv", "bin", "python");

    if (!fs.existsSync(pythonExe))
        throw new Error("hunyuan_t2i: venv not found. Reinstall the extension.");

    const workerScript = path.join(extDir, "t2i_worker.py");

    const modelsDir    = process.env.MODELS_DIR    || path.join(os.homedir(), ".modly", "models");
    const workspaceDir = context.workspaceDir      || path.join(os.homedir(), ".modly", "workspace");

    // text comes from connected Text node, params has the rest
    const prompt     = (input.text || "a beautiful landscape").trim();
    const paramsJson = JSON.stringify({ ...params, prompt });

    context.log(`T2I prompt: ${prompt}`);
    context.progress(2, "Starting worker...");

    return new Promise((resolve, reject) => {
        const worker = spawn(pythonExe, [
            workerScript,
            paramsJson,
            modelsDir,
            workspaceDir,
        ], {
            env: {
                ...process.env,
                MODELS_DIR:    modelsDir,
                WORKSPACE_DIR: workspaceDir,
                EXTENSION_DIR: extDir,
            },
        });

        let outputPath = null;
        let lineBuf    = "";

        worker.stdout.on("data", (chunk) => {
            lineBuf += chunk.toString();
            const lines = lineBuf.split("\n");
            lineBuf = lines.pop();
            for (const line of lines) {
                const trimmed = line.trim();
                if (!trimmed) continue;
                try {
                    const msg = JSON.parse(trimmed);
                    if      (msg.type === "progress") context.progress(msg.pct, msg.step || "");
                    else if (msg.type === "log")      context.log(msg.message || "");
                    else if (msg.type === "done")     outputPath = msg.output_path;
                    else if (msg.type === "error")    reject(new Error(msg.message || "Worker error"));
                } catch (_) {
                    context.log(`[worker] ${trimmed}`);
                }
            }
        });

        worker.stderr.on("data", (chunk) => {
            const text = chunk.toString().trim();
            if (text) context.log(`[stderr] ${text}`);
        });

        worker.on("error", (err) => reject(new Error(`Failed to start worker: ${err.message}`)));

        worker.on("close", (code) => {
            if (outputPath) {
                context.log(`Done: ${outputPath}`);
                resolve({ filePath: outputPath });
            } else if (code !== 0) {
                reject(new Error(`Worker exited with code ${code}. Check logs.`));
            } else {
                reject(new Error("Worker finished but returned no output path."));
            }
        });
    });
};

module.exports = processor;
