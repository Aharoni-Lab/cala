import getConfig from "./config";
import { add_on, pseudoSwitch } from "./switch";
import LineChart from "./components/lineChart";
import VideoPlayer from "./components/videoPlayer";
// import videojs from "video.js";
import "./vendor/htmx.2.0.6";
import "./css/video-js.css";
import "./css/grids.css";

window.add_on = add_on;
// window.videojs = videojs;
window.VideoPlayer = VideoPlayer;
window.LineChart = LineChart;
window.wsCallbacks = {};

document.addEventListener("DOMContentLoaded", async () => {
  const config = getConfig();
  console.log(config);

  const ws = new WebSocket(`ws://${window.location.host}/ws`);

  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log(data);
    pseudoSwitch(wsCallbacks, data.type_, data.payload);
  };
});
