// Canvas widget for the SCAIL-2 Identity Tracker node.
// Lets you draw ordered points / boxes per person on the reference image and the
// driving video's first frame. Placement order = identity (colour) order, matching
// the model palette in nodes_scail.py. Markers are serialised into the node's
// hidden "markers" STRING widget as {"reference":[...], "driving":[...]}.
//
// The frames appear after you press the node's play button (partial execution):
// the Python side returns them as `reference_preview` / `driving_preview` and we
// load them onto the canvas in onExecuted.

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Must match DEFAULT_PALETTE order in comfy_extras/nodes_scail.py
const PALETTE = ["#0000ff", "#ff0000", "#00ff00", "#ff00ff", "#00ffff", "#ffff00"];
const CANVAS_H = 320;
const HIT_PX = 12;

function viewURL(info) {
    const qs = `filename=${encodeURIComponent(info.filename)}&type=${info.type}` +
               `&subfolder=${encodeURIComponent(info.subfolder || "")}&rand=${Math.random()}`;
    const path = `/view?${qs}`;
    return (api && typeof api.apiURL === "function") ? api.apiURL(path) : path;
}

function setupNode(node) {
    const markersWidget = node.widgets?.find((w) => w.name === "markers");
    if (markersWidget) {
        markersWidget.hidden = true;
        markersWidget.computeSize = () => [0, -4];
        const origDraw = markersWidget.draw;
        markersWidget.draw = function () {}; // keep value, hide UI
        void origDraw;
    }

    const state = {
        side: "reference",
        mode: "box",
        markers: { reference: [], driving: [] },
        imgs: { reference: null, driving: null },
        view: { scale: 1, ox: 0, oy: 0 }, // transform for the active side
        drag: null,
    };
    node._scail = state;

    const syncFromWidget = () => {
        if (!markersWidget) return;
        try {
            const parsed = JSON.parse(markersWidget.value || "{}");
            state.markers.reference = Array.isArray(parsed.reference) ? parsed.reference : [];
            state.markers.driving = Array.isArray(parsed.driving) ? parsed.driving : [];
        } catch (e) { /* keep current */ }
    };
    syncFromWidget();

    const writeMarkers = () => {
        if (markersWidget) markersWidget.value = JSON.stringify(state.markers);
        node.graph?.setDirtyCanvas(true, true);
    };

    // --- DOM ---
    const container = document.createElement("div");
    container.style.cssText = "display:flex;flex-direction:column;gap:4px;width:100%;";

    const bar = document.createElement("div");
    bar.style.cssText = "display:flex;flex-wrap:wrap;gap:4px;font-size:11px;align-items:center;";
    container.appendChild(bar);

    const mkBtn = (label, on) => {
        const b = document.createElement("button");
        b.textContent = label;
        b.style.cssText = "padding:2px 6px;cursor:pointer;border-radius:4px;border:1px solid #555;background:#2a2a2a;color:#ddd;";
        b.onclick = (e) => { e.preventDefault(); e.stopPropagation(); on(b); redraw(); };
        bar.appendChild(b);
        return b;
    };

    const refBtn = mkBtn("Reference", () => (state.side = "reference"));
    const drvBtn = mkBtn("Driving", () => (state.side = "driving"));
    const sep = document.createElement("span"); sep.style.cssText = "width:8px;"; bar.appendChild(sep);
    const boxBtn = mkBtn("Box", () => (state.mode = "box"));
    const ptBtn = mkBtn("Point", () => (state.mode = "point"));
    const sep2 = document.createElement("span"); sep2.style.cssText = "width:8px;"; bar.appendChild(sep2);
    mkBtn("Undo", () => state.markers[state.side].pop() && writeMarkers());
    mkBtn("Clear", () => { state.markers[state.side] = []; writeMarkers(); });

    const hint = document.createElement("div");
    hint.style.cssText = "font-size:10px;color:#999;";
    container.appendChild(hint);

    const canvas = document.createElement("canvas");
    canvas.style.cssText = "width:100%;height:" + CANVAS_H + "px;background:#1a1a1a;border-radius:4px;display:block;cursor:crosshair;";
    container.appendChild(canvas);
    const ctx = canvas.getContext("2d");

    // --- drawing ---
    const activeImg = () => state.imgs[state.side];
    const activeMarks = () => state.markers[state.side];

    function computeView(img) {
        const cw = canvas.width, ch = canvas.height;
        const scale = Math.min(cw / img.naturalWidth, ch / img.naturalHeight);
        state.view = {
            scale,
            ox: (cw - img.naturalWidth * scale) / 2,
            oy: (ch - img.naturalHeight * scale) / 2,
        };
    }

    const imgToScreen = (x, y) => [state.view.ox + x * state.view.scale, state.view.oy + y * state.view.scale];
    function screenToImg(clientX, clientY) {
        const r = canvas.getBoundingClientRect();
        const cx = (clientX - r.left) * (canvas.width / r.width);
        const cy = (clientY - r.top) * (canvas.height / r.height);
        return [(cx - state.view.ox) / state.view.scale, (cy - state.view.oy) / state.view.scale];
    }

    function redraw() {
        // size backing store to displayed size
        const w = canvas.clientWidth || node.size[0] - 20;
        if (canvas.width !== w) canvas.width = w;
        if (canvas.height !== CANVAS_H) canvas.height = CANVAS_H;
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        for (const b of [refBtn, drvBtn]) b.style.outline = "";
        (state.side === "reference" ? refBtn : drvBtn).style.outline = "2px solid #6cf";
        for (const b of [ptBtn, boxBtn]) b.style.outline = "";
        (state.mode === "point" ? ptBtn : boxBtn).style.outline = "2px solid #6cf";

        const img = activeImg();
        if (!img) {
            ctx.fillStyle = "#777";
            ctx.font = "12px sans-serif";
            ctx.fillText("Press the node's ▶ play button to load this frame", 12, 24);
            hint.textContent = "Order you place markers = colour order. Right-click a marker to delete.";
            return;
        }
        computeView(img);
        ctx.drawImage(img, state.view.ox, state.view.oy, img.naturalWidth * state.view.scale, img.naturalHeight * state.view.scale);

        const marks = activeMarks();
        marks.forEach((m, i) => {
            const color = PALETTE[i % PALETTE.length];
            ctx.lineWidth = 2;
            ctx.strokeStyle = color;
            ctx.fillStyle = color;
            if (m.type === "box") {
                const [sx, sy] = imgToScreen(m.x, m.y);
                ctx.strokeRect(sx, sy, m.w * state.view.scale, m.h * state.view.scale);
                drawLabel(i + 1, color, sx + 2, sy + 2);
            } else {
                const [sx, sy] = imgToScreen(m.x, m.y);
                ctx.beginPath(); ctx.arc(sx, sy, 5, 0, Math.PI * 2); ctx.fill();
                ctx.strokeStyle = "#000"; ctx.stroke();
                drawLabel(i + 1, color, sx + 7, sy - 7);
            }
        });

        if (state.drag && state.drag.moved && state.mode === "box") {
            const [sx, sy] = imgToScreen(state.drag.x0, state.drag.y0);
            ctx.setLineDash([4, 3]);
            ctx.strokeStyle = PALETTE[marks.length % PALETTE.length];
            ctx.strokeRect(sx, sy, (state.drag.x1 - state.drag.x0) * state.view.scale, (state.drag.y1 - state.drag.y0) * state.view.scale);
            ctx.setLineDash([]);
        }
        hint.textContent = `${state.side}: ${marks.length} marker(s). Order = colour order. Right-click to delete.`;
    }

    function drawLabel(n, color, x, y) {
        ctx.font = "bold 13px sans-serif";
        ctx.fillStyle = "#000";
        ctx.fillText(String(n), x + 1, y + 13 + 1);
        ctx.fillStyle = color;
        ctx.fillText(String(n), x, y + 13);
    }
    node._scailRedraw = redraw;

    // --- interaction ---
    canvas.addEventListener("pointerdown", (e) => {
        if (e.button !== 0 || !activeImg()) return;
        const [ix, iy] = screenToImg(e.clientX, e.clientY);
        state.drag = { x0: ix, y0: iy, x1: ix, y1: iy, moved: false };
        canvas.setPointerCapture(e.pointerId);
    });
    canvas.addEventListener("pointermove", (e) => {
        if (!state.drag) return;
        const [ix, iy] = screenToImg(e.clientX, e.clientY);
        state.drag.x1 = ix; state.drag.y1 = iy;
        if (Math.hypot(ix - state.drag.x0, iy - state.drag.y0) * state.view.scale > 3) state.drag.moved = true;
        redraw();
    });
    canvas.addEventListener("pointerup", (e) => {
        if (!state.drag || !activeImg()) { state.drag = null; return; }
        const d = state.drag; state.drag = null;
        const marks = activeMarks();
        if (state.mode === "box" && d.moved) {
            const x = Math.min(d.x0, d.x1), y = Math.min(d.y0, d.y1);
            const w = Math.abs(d.x1 - d.x0), h = Math.abs(d.y1 - d.y0);
            if (w > 2 && h > 2) marks.push({ type: "box", x, y, w, h });
        } else if (state.mode === "point") {
            marks.push({ type: "point", x: d.x0, y: d.y0 });
        }
        writeMarkers(); redraw();
    });
    canvas.addEventListener("contextmenu", (e) => {
        e.preventDefault();
        if (!activeImg()) return;
        const r = canvas.getBoundingClientRect();
        const px = (e.clientX - r.left) * (canvas.width / r.width);
        const py = (e.clientY - r.top) * (canvas.height / r.height);
        const marks = activeMarks();
        let best = -1, bestD = HIT_PX;
        marks.forEach((m, i) => {
            let cx, cy;
            if (m.type === "box") { [cx, cy] = imgToScreen(m.x + m.w / 2, m.y + m.h / 2); }
            else { [cx, cy] = imgToScreen(m.x, m.y); }
            const dist = Math.hypot(px - cx, py - cy);
            if (dist < bestD) { bestD = dist; best = i; }
        });
        if (best >= 0) { marks.splice(best, 1); writeMarkers(); redraw(); }
    });

    const widget = node.addDOMWidget("scail_canvas", "scail_canvas", container, { serialize: false, hideOnZoom: false });
    widget.computeSize = function () { return [node.size[0], CANVAS_H + 60]; };

    // load previews pushed by the Python node on execution
    node._scailOnExecuted = (message) => {
        const load = (side, info) => {
            if (!info) return;
            const im = new Image();
            im.onload = () => { state.imgs[side] = im; if (state.side === side) redraw(); };
            im.src = viewURL(info);
        };
        load("reference", message?.reference_preview?.[0]);
        load("driving", message?.driving_preview?.[0]);
    };

    node._scailSync = () => { syncFromWidget(); redraw(); };

    if (node.size[1] < CANVAS_H + 140) node.size[1] = CANVAS_H + 140;
    setTimeout(redraw, 50);
}

app.registerExtension({
    name: "scail.identityTracker",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "SCAIL2IdentityTracker") return;

        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onCreated?.apply(this, arguments);
            setupNode(this);
            return r;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);
            this._scailOnExecuted?.(message);
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = onConfigure?.apply(this, arguments);
            this._scailSync?.();
            return r;
        };
    },
});
