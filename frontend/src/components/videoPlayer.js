import videojs from 'video.js';

class VideoPlayer {
    constructor(elementId, options = {}) {
        this.elementId = elementId;
        this.defaultOptions = {
            fluid: true,
            liveui: true
        };
        this.options = {...this.defaultOptions, ...options};
        this.player = null;
    }

    initialize() {
        this.player = videojs(this.elementId, this.options);
        return this.player;
    }

    play() {
        if (this.player) {
            this.player.play();
        }
    }

    destroy() {
        if (this.player) {
            this.player.dispose();
        }
    }
}

export default VideoPlayer;
