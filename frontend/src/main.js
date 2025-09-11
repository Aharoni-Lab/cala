import VideoPlayer from "./components/videoPlayer";
import videojs from "video.js";
import "./vendor/htmx.2.0.6";
import "./css/video-js.css";
import "./css/grids.css";

window.videojs = videojs;
window.VideoPlayer = VideoPlayer;

function getCookie(name) {
  const value = `; ${document.cookie}`;
  const parts = value.split(`; ${name}=`);
  if (parts.length === 2) return parts.pop().split(";").shift();
}

function getConfig() {
  let config_byt = getCookie("config");
  const config_str = atob(config_byt);
  return JSON.parse(config_str);
}

document.addEventListener("DOMContentLoaded", async () => {
  const config = getConfig();

  // // Create WebSocket connection
  const ws = new WebSocket(`ws://${window.location.host}/ws`);
  //
  // const chart = new LineChart('#plot-container', config.metric_plot);
  // await chart.initialize();
  //
  // const counter = new FrameNumber('frame-index');
  // counter.initialize();
  //
  // ws.onmessage = (event) => {
  //     const data = JSON.parse(event.data);
  //
  //     switch (data.payload.type_) {
  //         case "frame_index":
  //             delete data.payload.type_;
  //             counter.updateData(data.payload);
  //             break;
  //         case "component_count":
  //             delete data.payload.type_;
  //             chart.updateData(data.payload);
  //             break;
  //         default:
  //             console.log("Nothing fits! Data:", data);
  //             break;
  //     }
  // };
});
