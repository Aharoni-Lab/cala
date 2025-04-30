import VideoPlayer from './components/videoPlayer';
import LineChart from './components/lineChart';
import WebSocketClient from './utils/webSocket';

// Initialize application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    const config = window.config || {};

    // Initialize video player
    const videoPlayer = new VideoPlayer('stream-player', {
        fluid: true,
        liveui: true
    });
    videoPlayer.initialize();
    videoPlayer.play();

    // Initialize chart
    const chart = new LineChart('#plot-container', {
        width: config.video?.width,
        height: config.video?.height
    });

    // Setup WebSocket and connect chart to data stream
    const wsUrl = `ws://${window.location.host}/ws`;
    const wsClient = new WebSocketClient(wsUrl);

    // Initialize chart then connect to WebSocket
    chart.initialize().then(() => {
        wsClient.addDataListener(data => {
            chart.updateData(data);
        });
        wsClient.connect();
    });
});
