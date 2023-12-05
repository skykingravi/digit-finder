import { weight0, weight1, weight2 } from "./assets/weights.js";
import { bias0, bias1, bias2 } from "./assets/biases.js";

const init = () => {
    // Text & Buttons
    const clearCanvasBtn = document.getElementById("clearCanvasBtn");

    // Get the canvas element and its context
    var canvas = document.getElementById("drawingCanvas");
    var context = canvas.getContext("2d", { willReadFrequently: true });

    // Set line properties
    context.lineCap = "round";
    context.lineWidth = 40;
    context.strokeStyle = "#5EDB88";

    // Variables to track drawing state
    var isDrawing = false;

    // Check if the device supports touch events
    const isTouchDevice =
        "ontouchstart" in window ||
        navigator.maxTouchPoints > 0 ||
        navigator.msMaxTouchPoints > 0;

    function getEventCoordinates(event) {
        // Check if the event is a touch event or a mouse event
        if (isTouchDevice && event.type.startsWith("touch")) {
            // Touch event
            const touch = event.touches[0];
            return { clientX: touch.clientX, clientY: touch.clientY };
        } else {
            // Mouse event
            return { clientX: event.clientX, clientY: event.clientY };
        }
    }

    // Function to get mouse position relative to the canvas
    function getPosition(e) {
        const { clientX, clientY } = getEventCoordinates(e);
        var rect = canvas.getBoundingClientRect();
        var X = clientX - rect.left;
        var Y = clientY - rect.top;
        return { x: X, y: Y };
    }

    // Function to start drawing
    function startDrawing(e) {
        e.preventDefault();
        isDrawing = true;
        context.beginPath();
        draw(e);
    }

    // Function to stop drawing
    function stopDrawing() {
        isDrawing = false;
        context.beginPath();
        predictDigit();
    }

    // Function to draw on the canvas
    function draw(e) {
        if (!isDrawing) return;

        // Get the correct event position based on the event type
        const { x, y } = getPosition(e);
        context.lineTo(x, y);
        context.stroke();
        context.beginPath();
        context.moveTo(x, y);
    }

    // Set up event listeners for both mouse and touch events
    canvas.addEventListener("mousedown", startDrawing);
    canvas.addEventListener("mousemove", draw);
    canvas.addEventListener("mouseup", stopDrawing);
    canvas.addEventListener("mouseleave", stopDrawing);
    canvas.addEventListener("touchstart", startDrawing, { passive: false });
    canvas.addEventListener("touchmove", draw, { passive: false });
    canvas.addEventListener("touchend", stopDrawing);

    // Clear the canvas
    function clearCanvas() {
        context.clearRect(0, 0, canvas.width, canvas.height);
        predictDigit();
    }

    // ReLU Activation Function
    function relu(arr) {
        return arr.map((value) => Math.max(0, value));
    }

    // Softmax Activation Function
    function softmax(arr) {
        // Calculate the maximum value in the array
        const maxVal = Math.max(...arr);

        // Subtract the maximum value from each element to avoid numerical instability
        const expArr = arr.map((x) => Math.exp(x - maxVal));

        // Calculate the sum of the exponential values
        const expSum = expArr.reduce((sum, value) => sum + value, 0);

        // Calculate the softmax values
        const softmaxArr = expArr.map((value) => value / expSum);

        return softmaxArr;
    }

    function GetResizedCanvas(sourceCanvas, targetSize) {
        var canvas1 = sourceCanvas;

        while (1) {
            if (canvas1.width > targetSize * 2) {
                var canvas2 = document.createElement("canvas");
                canvas2.width = canvas1.width * 0.5;
                canvas2.height = canvas1.height * 0.5;

                var canvas2Context = canvas2.getContext("2d");
                canvas2Context.drawImage(
                    canvas1,
                    0,
                    0,
                    canvas2.width,
                    canvas2.height
                );

                canvas1 = canvas2;
            } else {
                var canvas2 = document.createElement("canvas");
                canvas2.width = targetSize;
                canvas2.height = targetSize;

                var canvas2Context = canvas2.getContext("2d");
                canvas2Context.drawImage(
                    canvas1,
                    0,
                    0,
                    canvas2.width,
                    canvas2.height
                );

                return canvas2;
            }
        }
    }

    // Prediction
    const predictDigit = () => {
        // Resized input
        const newCanvas = GetResizedCanvas(canvas, 28);
        var context1 = newCanvas.getContext("2d");
        var imageData = context1.getImageData(
            0,
            0,
            newCanvas.width,
            newCanvas.height
        );

        // Intensity values
        var pixelValues = [];
        for (let i = 0; i < 3136; i += 4) {
            pixelValues.push(imageData.data[i + 3] / 255);
        }

        // Output-1 -> o1 = relu(sum(x*w) + b)
        var op1 = new Array(128);
        for (let i = 0; i < 128; i++) {
            var value = bias0[i];
            for (let j = 0; j < 784; j++) {
                value += pixelValues[j] * weight0[j][i];
            }
            op1[i] = value;
        }
        op1 = relu(op1);

        // Output-2 -> o2 = relu(sum(o1*w) + b)
        var op2 = new Array(32);
        for (let i = 0; i < 32; i++) {
            var value = bias1[i];
            for (let j = 0; j < 128; j++) {
                value += op1[j] * weight1[j][i];
            }
            op2[i] = value;
        }
        op2 = relu(op2);

        // Output-3 -> o3 = softmax(sum(o2*w) + b)
        var op3 = new Array(10);
        for (let i = 0; i < 10; i++) {
            var value = bias2[i];
            for (let j = 0; j < 32; j++) {
                value += op2[j] * weight2[j][i];
            }
            op3[i] = value;
        }
        op3 = softmax(op3);

        // Display the predictions
        for (let i = 0; i < 10; i++) {
            document.getElementsByClassName(`bar${i}`)[0].style.height =
                op3[i] * 100 > 1 ? `${op3[i] * 100}%` : "3%";
            document.getElementsByClassName(
                `bar${i}`
            )[0].title = `Prob: ${op3[i]}`;
        }
    };

    predictDigit();

    // Interactions
    clearCanvasBtn.addEventListener("click", clearCanvas);
};

document.addEventListener("DOMContentLoaded", init);
