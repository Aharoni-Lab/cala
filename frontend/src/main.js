import VideoPlayer from "./components/videoPlayer";
import LineChart from "./components/lineChart";


document.addEventListener('DOMContentLoaded', () => {
    const config = window.config

    // Initialize video player
    const videoPlayer = new VideoPlayer('stream-player', {
        fluid: true,
        liveui: true
    });
    videoPlayer.initialize();
    videoPlayer.play();

    const chart = new LineChart('#plot-container', config.plot)

    // Create WebSocket connection
    const ws = new WebSocket(`ws://${window.location.host}/ws`);

    chart.initialize().then(() => {
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            chart.updateData(data);
        };
    });
});
