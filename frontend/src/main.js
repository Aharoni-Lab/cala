import VideoPlayer from "./components/videoPlayer";
import LineChart from "./components/lineChart";
import FrameNumber from "./components/frameNumber";
import './vendor/htmx.2.0.6';
import './css/video-js.css';

function getCookie(name) {
    const value = `; ${document.cookie}`;
    console.log("cookie value", value);
    const parts = value.split(`; ${name}=`);
    if (parts.length === 2) return parts.pop().split(';').shift();
}

function getConfig() {
    let config_str = getCookie('config');
    console.log("config string", config_str);
    let config = JSON.parse(config_str);
    console.log(config);
    return config;
}

document.addEventListener('DOMContentLoaded', async () => {
    const config = getConfig()

    // Initialize video player
    const videoPlayer = new VideoPlayer('stream-player', {
        fluid: true,
        liveui: true
    });
    videoPlayer.initialize();
    videoPlayer.play();

    // Initialize video player
    const footprintPlayer = new VideoPlayer('footprint-player', {
        fluid: true,
        liveui: true
    });
    footprintPlayer.initialize();
    footprintPlayer.play();

    // Create WebSocket connection
    const ws = new WebSocket(`ws://${window.location.host}/ws`);

    const chart = new LineChart('#plot-container', config.metric_plot);
    await chart.initialize();

    const counter = new FrameNumber('frame-index');
    counter.initialize();

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);

        switch (data.payload.type_) {
            case "frame_index":
                delete data.payload.type_;
                counter.updateData(data.payload);
                break;
            case "component_count":
                delete data.payload.type_;
                chart.updateData(data.payload);
                break;
            default:
                console.log("Nothing fits! Data:", data);
                break;
        }
    };
});
