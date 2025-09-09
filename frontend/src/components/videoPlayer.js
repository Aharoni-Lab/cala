import videojs from "video.js";

class VideoPlayer {
  constructor(elementId, options = {}) {
    this.elementId = elementId;
    this.defaultOptions = {
      fluid: true,
      liveui: true,
    };
    this.options = { ...this.defaultOptions, ...options };
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

  buildElement(name) {
    let elem = document.createElement("div");
    elem.setAttribute("class", "video-container gfg");
    let h3 = document.createElement("h3");
    h3.innerHTML = name;
    let vid = document.createElement("video");
    vid.setAttribute("id", name);
    vid.setAttribute("class", "video_js");
    vid.setAttribute("data-setup", '{"fluid": true}');
    vid.setAttribute("controls", "");
    let src = document.createElement("source");
    src.setAttribute("src", name.concat("/stream.m3u8"));
    src.setAttribute("type", "application/x-mpegURL");
    vid.appendChild(src);
    console.log(vid);
    console.log(vid.children);
    elem.append(h3, vid);
    console.log(elem.children); // why can't it add video like a normal elem?
    return elem;
  }
}
export default VideoPlayer;
