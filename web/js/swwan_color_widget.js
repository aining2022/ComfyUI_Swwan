// SPDX-License-Identifier: MIT
// Independent Swwan implementation; see licenses/MIT-Swwan.txt.
import { app } from "../../scripts/app.js";

// A custom widget keeps COLORCODE sockets independent of other plugins. Use a
// custom type so current LiteGraph dispatches pointerdown to the color picker.
let activePicker;

function colorWidget(name, initial) {
    return {
        name,
        type: "COLORCODE",
        value: initial,
        options: { widgetType: "COLORCODE" },
        computeSize: () => [180, 24],
        serializeValue() { return this.value; },
        draw(ctx, node, width, y, height) {
            if (this.hidden || this.type !== "COLORCODE") return;
            ctx.save();
            ctx.fillStyle = "#252525";
            ctx.fillRect(10, y, width - 20, height);
            ctx.fillStyle = /^#[0-9a-f]{6}$/i.test(this.value) ? this.value : "#364254";
            ctx.fillRect(width - 36, y + 4, 18, height - 8);
            ctx.fillStyle = "#ddd";
            ctx.font = "12px sans-serif";
            ctx.textAlign = "left";
            ctx.fillText(`${this.name}: ${this.value}`, 16, y + height * 0.7, width - 58);
            ctx.restore();
        },
        mouse(event, pos, node) {
            if (event.type !== "pointerdown" || this.hidden) return false;
            activePicker?.();
            const picker = document.createElement("input");
            picker.type = "color";
            picker.value = /^#[0-9a-f]{6}$/i.test(this.value) ? this.value : "#364254";
            picker.style.position = "fixed";
            picker.style.left = "-9999px";
            document.body.appendChild(picker);
            const remove = () => {
                picker.remove();
                document.removeEventListener("pointerdown", remove, true);
                if (activePicker === remove) activePicker = undefined;
            };
            activePicker = remove;
            // Native picker cancellation differs between browsers. The next
            // page click also cleans up a picker that closed without change.
            document.addEventListener("pointerdown", remove, { once: true, capture: true });
            picker.addEventListener("input", () => {
                this.value = picker.value;
                this.callback?.(this.value);
                if (node.graph) node.graph._version++;
                node.setDirtyCanvas?.(true, true);
            });
            picker.addEventListener("change", remove, { once: true });
            picker.addEventListener("cancel", remove, { once: true });
            picker.addEventListener("blur", remove, { once: true });
            picker.click();
            return true;
        },
    };
}

app.registerExtension({
    name: "Swwan.ColorCodeWidget",
    getCustomWidgets() {
        return {
            COLORCODE(node, inputName, inputData) {
                return {
                    widget: node.addCustomWidget(colorWidget(inputName, inputData?.[1]?.default || "#364254")),
                    minWidth: 180,
                    minHeight: 24,
                };
            },
        };
    },
});
