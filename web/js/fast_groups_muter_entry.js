/**
 * Fast Groups Muter - 移植自 rgthree-comfy
 * 简化版入口文件，减少依赖
 */

import { app } from "../../scripts/app.js";

const PROPERTY_SORT = "sort";
const PROPERTY_SORT_CUSTOM_ALPHA = "customSortAlphabet";
const PROPERTY_MATCH_COLORS = "matchColors";
const PROPERTY_MATCH_TITLE = "matchTitle";
const PROPERTY_SHOW_NAV = "showNav";
const PROPERTY_SHOW_ALL_GRAPHS = "showAllGraphs";
const PROPERTY_RESTRICTION = "toggleRestriction";

// 简化的 Fast Groups Service
class FastGroupsService {
    constructor() {
        this.fastGroupNodes = [];
        this.groupsCache = [];
        this.lastUpdate = 0;
    }
    
    addFastGroupNode(node) {
        this.fastGroupNodes.push(node);
        this.scheduleRefresh();
    }
    
    removeFastGroupNode(node) {
        const index = this.fastGroupNodes.indexOf(node);
        if (index > -1) {
            this.fastGroupNodes.splice(index, 1);
        }
    }
    
    scheduleRefresh() {
        setTimeout(() => {
            for (const node of this.fastGroupNodes) {
                node.refreshWidgets && node.refreshWidgets();
            }
        }, 100);
    }
    
    getGroups(sort = "position") {
        const now = Date.now();
        if (now - this.lastUpdate < 400 && this.groupsCache.length) {
            return this.groupsCache;
        }
        
        const graph = app.canvas.getCurrentGraph() || app.graph;
        this.groupsCache = [...(graph._groups || [])];
        
        // 计算每个组内的节点
        for (const group of this.groupsCache) {
            group.recomputeInsideNodes();
            const groupNodes = Array.from(group._children).filter(c => c instanceof LGraphNode);
            group.rgthree_hasAnyActiveNode = groupNodes.some(n => n.mode === LiteGraph.ALWAYS);
        }
        
        // 排序
        if (sort === "alphanumeric") {
            this.groupsCache.sort((a, b) => a.title.localeCompare(b.title));
        } else if (sort === "position") {
            this.groupsCache.sort((a, b) => {
                const aY = Math.floor(a._pos[1] / 30);
                const bY = Math.floor(b._pos[1] / 30);
                if (aY === bY) {
                    return Math.floor(a._pos[0] / 30) - Math.floor(b._pos[0] / 30);
                }
                return aY - bY;
            });
        }
        
        this.lastUpdate = now;
        return this.groupsCache;
    }
}

const SERVICE = new FastGroupsService();

// Fast Groups Muter 节点
class FastGroupsMuter {
    constructor() {
        this.type = "Fast Groups Muter (rgthree)";
        this.title = "Fast Groups Muter";
        this.size = [300, 100];
        this.properties = {
            [PROPERTY_MATCH_COLORS]: "",
            [PROPERTY_MATCH_TITLE]: "",
            [PROPERTY_SHOW_NAV]: true,
            [PROPERTY_SHOW_ALL_GRAPHS]: true,
            [PROPERTY_SORT]: "position",
            [PROPERTY_SORT_CUSTOM_ALPHA]: "",
            [PROPERTY_RESTRICTION]: "default"
        };
        this.widgets = [];
        this.serialize_widgets = false;
        this.isVirtualNode = true;
        
        this.addOutput("OPT_CONNECTION", "*");
    }
    
    onAdded() {
        SERVICE.addFastGroupNode(this);
    }
    
    onRemoved() {
        SERVICE.removeFastGroupNode(this);
    }
    
    refreshWidgets() {
        const groups = SERVICE.getGroups(this.properties[PROPERTY_SORT]);
        const filterColors = (this.properties[PROPERTY_MATCH_COLORS] || "")
            .split(",")
            .map(c => c.trim())
            .filter(c => c);
        
        let index = 0;
        for (const group of groups) {
            // 颜色过滤
            if (filterColors.length) {
                let groupColor = (group.color || "").replace("#", "").toLowerCase();
                if (!groupColor || !filterColors.some(fc => groupColor.includes(fc.replace("#", "")))) {
                    continue;
                }
            }
            
            // 标题过滤
            if (this.properties[PROPERTY_MATCH_TITLE]) {
                try {
                    if (!new RegExp(this.properties[PROPERTY_MATCH_TITLE], "i").test(group.title)) {
                        continue;
                    }
                } catch (e) {
                    continue;
                }
            }
            
            const widgetName = `enable_${group.title}`;
            let widget = this.widgets.find(w => w.name === widgetName);
            
            if (!widget) {
                widget = {
                    name: widgetName,
                    type: "toggle",
                    value: group.rgthree_hasAnyActiveNode || false,
                    group: group,
                    callback: (value) => {
                        const groupNodes = Array.from(group._children).filter(c => c instanceof LGraphNode);
                        const newMode = value ? LiteGraph.ALWAYS : LiteGraph.NEVER;
                        for (const node of groupNodes) {
                            node.mode = newMode;
                        }
                        group.rgthree_hasAnyActiveNode = value;
                        app.graph.setDirtyCanvas(true, false);
                    }
                };
                this.addCustomWidget(widget);
            }
            
            widget.value = group.rgthree_hasAnyActiveNode || false;
            
            if (this.widgets[index] !== widget) {
                const oldIndex = this.widgets.indexOf(widget);
                if (oldIndex > -1) {
                    this.widgets.splice(oldIndex, 1);
                    this.widgets.splice(index, 0, widget);
                }
            }
            
            index++;
        }
        
        // 移除多余的 widgets
        while (this.widgets.length > index) {
            this.widgets.pop();
        }
        
        this.setSize(this.computeSize());
        app.graph.setDirtyCanvas(true, true);
    }
    
    getExtraMenuOptions(canvas, options) {
        options.push(
            null,
            {
                content: "Mute all",
                callback: () => {
                    for (const widget of this.widgets) {
                        widget.callback && widget.callback(false);
                    }
                }
            },
            {
                content: "Enable all",
                callback: () => {
                    for (const widget of this.widgets) {
                        widget.callback && widget.callback(true);
                    }
                }
            },
            {
                content: "Toggle all",
                callback: () => {
                    for (const widget of this.widgets) {
                        widget.callback && widget.callback(!widget.value);
                    }
                }
            }
        );
    }
}

// 注册节点
app.registerExtension({
    name: "Comfy.FastGroupsMuter",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "Fast Groups Muter (rgthree)") {
            nodeType.prototype = Object.assign(
                Object.create(LGraphNode.prototype),
                FastGroupsMuter.prototype
            );
            nodeType.prototype.constructor = nodeType;
        }
    },
    async nodeCreated(node) {
        if (node.type === "Fast Groups Muter (rgthree)") {
            setTimeout(() => node.refreshWidgets && node.refreshWidgets(), 100);
        }
    }
});
