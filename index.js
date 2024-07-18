const GRAVITY = 0.1;

// Used by project buttons
function gotoUrl(url)
{
	window.location.assign(url);
}

const canvas = new Canvas("header-canvas", new Colour(0, 0, 0, 1));
const fireworkSpawner = new FireworkLauncher(canvas, 1, 10, 45, GRAVITY, 10, 40, 10, 30, 10, 20);
canvas.addSpawner(fireworkSpawner);
canvas.updateDimensions();
canvas.start();

window.onresize = (e) => canvas.updateDimensions();
