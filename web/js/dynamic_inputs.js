import { app } from "../../scripts/app.js";

const DYNAMIC_IMAGE_NODES = new Set([
    "CrossFadeImagesMulti",
    "TransitionImagesMulti",
    "ImageBatchMulti",
    "ImageAddMulti",
    "ImageConcatMulti",
]);

function getPrefixIndex(name, prefix) {
    const match = name?.match(new RegExp(`^${prefix}(\\d+)$`));
    return match ? Number(match[1]) : null;
}

function setupDynamicInputs(node, { type, prefix, countWidget = "inputcount", slotOptions } = {}) {
    const rebuild = (value) => {
        if (!node.inputs) {
            node.inputs = [];
        }

        const countWidgetObj = node.widgets?.find((widget) => widget.name === countWidget);
        const target = Math.floor(Number(value ?? countWidgetObj?.value));
        if (!Number.isFinite(target) || target < 1) {
            return;
        }

        const matchingInputs = node.inputs
            .map((input, index) => ({ input, index, number: getPrefixIndex(input.name, prefix) }))
            .filter(({ number }) => number !== null);
        const current = matchingInputs.length;

        if (target === current) {
            return;
        }

        if (target < current) {
            matchingInputs
                .filter(({ number }) => number > target)
                .sort((a, b) => b.index - a.index)
                .forEach(({ index }) => node.removeInput(index));
        } else {
            for (let index = current + 1; index <= target; index++) {
                node.addInput(`${prefix}${index}`, type, slotOptions);
            }
        }

        const computedSize = node.computeSize?.();
        if (computedSize && node.size) {
            node.setSize?.([node.size[0], computedSize[1]]);
        } else {
            node.graph?.setDirtyCanvas?.(true, true);
        }
    };

    const countWidgetObj = node.widgets?.find((widget) => widget.name === countWidget);
    if (countWidgetObj) {
        const originalCallback = countWidgetObj.callback;
        countWidgetObj.callback = function (value) {
            const result = originalCallback?.apply(this, arguments);
            rebuild(value);
            return result;
        };
    }

    const originalOnConfigure = node.onConfigure;
    node.onConfigure = function () {
        const result = originalOnConfigure?.apply(this, arguments);
        rebuild();
        return result;
    };
};

app.registerExtension({
    name: "Swwan.DynamicInputs",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!nodeData?.category?.startsWith("Swwan") || !DYNAMIC_IMAGE_NODES.has(nodeData.name)) {
            return;
        }

        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = originalOnNodeCreated?.apply(this, arguments);
            setupDynamicInputs(this, { type: "IMAGE", prefix: "image_", slotOptions: { shape: 7 } });
            return result;
        };
    },
});
