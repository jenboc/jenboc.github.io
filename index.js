const canvas = document.getElementById("header-canvas");
const ctx = canvas.getContext("2d");
const header = document.getElementById("header");

var width;
var height;

function setCanvasSize(canvasCtx) 
{
	const headerRect = header.getBoundingClientRect();
	height = canvasCtx.height = headerRect.top + headerRect.height;
	width = canvasCtx.width = headerRect.width;
}

setCanvasSize(canvas);
ctx.fillRect(0, 0, width, height);
