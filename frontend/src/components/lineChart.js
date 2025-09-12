import embed from "vega-embed";
import * as vega from "vega";

class LineChart {
  constructor(containerId, config = {}) {
    this.containerId = containerId;
    this.config = config;
    this.view = null;
  }

  createSpec() {
    return {
      $schema: "https://vega.github.io/schema/vega-lite/v6.json",
      title: {
        text: "Frame Index vs. Component Count",
        fontSize: 20,
      },
      description: "Livestream",
      width: this.config.width,
      height: this.config.height,
      // autosize: {
      //     type: "fit",
      //     contains: "padding"
      // },
      data: { name: "table" },
      mark: "line",
      encoding: {
        x: {
          field: "index",
          type: "quantitative",
          title: "Frame Index",
        },
        y: { field: "count", type: "quantitative", title: "Count" },
      },
    };
  }

  async initialize() {
    const embedding = await embed(this.containerId, this.createSpec());
    this.view = embedding.view;
    return this;
  }

  updateData(data) {
    if (!this.view) return;

    const currentData = this.view.data("table");
    let changeSet;

    if (currentData.length >= 100) {
      // Get the oldest time we want to remove
      const oldestIdx = currentData[0].index;
      changeSet = vega
        .changeset()
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
