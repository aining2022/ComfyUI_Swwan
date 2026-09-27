// SPDX-License-Identifier: MIT
// Independent Swwan implementation; see licenses/MIT-Swwan.txt.
import { app } from '../../scripts/app.js';
app.registerExtension({
    name: 'Swwan.FastPreview',
    beforeRegisterNodeDef(Node, data) {
        if (data.name !== 'SwwanFastPreview') return;
        const executed = Node.prototype.onExecuted;
        Node.prototype.onExecuted = function(message) {
            const result = executed?.apply(this, arguments);
            if (message.bg_image?.length) {
                const image = new Image();
                image.onload = () => {this.imgs = [image]; this.setDirtyCanvas?.(true, true);};
                const base64 = message.bg_image[0];
                const format = base64.startsWith('iVBOR') ? 'png' : base64.startsWith('UklGR') ? 'webp' : 'jpeg';
                image.src = `data:image/${format};base64,${base64}`;
            }
            return result;
        };
    },
});
