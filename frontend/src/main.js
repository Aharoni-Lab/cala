import VideoPlayer from "./components/videoPlayer";
import LineChart from "./components/lineChart";
import FrameNumber from "./components/frameNumber";
import './css/video-js.css';

document.addEventListener('DOMContentLoaded', async () => {
    const config = window.config


    // Initialize video player
    const videoPlayer = new VideoPlayer('stream-player', {
        fluid: true,
        liveui: true
    });
    videoPlayer.initialize();
    videoPlayer.play();

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
