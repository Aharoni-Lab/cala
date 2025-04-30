import embed from 'vega-embed';
import * as vega from 'vega';

class LineChart {
    constructor(containerId, config = {}) {
        this.containerId = containerId;
        this.config = config;
        this.view = null;
        this.maxPoints = 100;
    }

    createSpec() {
        return {
            $schema: "https://vega.github.io/schema/vega-lite/v5.json",
            description: "Live data stream",
            width: this.config.width || 640,
            height: this.config.height || 480,
            data: {name: "table"},
            mark: "line",
            encoding: {
                x: {
                    field: "index",
                    type: "quantitative",
                },
                y: {field: "value", type: "quantitative"},
            }
        };
    }

    async initialize() {
        const result = await embed(this.containerId, this.createSpec());
        this.view = result.view;
        return this;
    }

    updateData(data) {
        if (!this.view) return;

        const currentData = this.view.data("table");
        let changeSet;

        if (currentData.length >= this.maxPoints) {
            // Get the oldest time we want to remove
            const oldestIdx = currentData[0].index;
            changeSet = vega.changeset()
                .insert(data)
                .remove((t) => t.index === oldestIdx);
        } else {
            // Just insert if we're under maxPoints
            changeSet = vega.changeset().insert(data);
        }

        this.view.change("table", changeSet).run();
    }
}

export default LineChart;
