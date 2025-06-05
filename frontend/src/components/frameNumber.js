class FrameNumber {
    constructor(elementId) {
        this.elementId = elementId;
    }

    initialize() {
        this.liveNumberElement = document.getElementById(this.elementId);

    }

    updateData(data) {
        this.liveNumberElement.textContent = data.index.toString();
    }
}

export default FrameNumber;
