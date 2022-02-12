class Object {
	constructor(objId="none", rotateDeg=0, innerText="none", fgcolor="#000000", bgcolor="#ffffff", classes="", posLeft="none", posTop="none") {
		if (objId == "none") {
			this.object = document.createElement("div")
		}
		else {
			this.object = document.getElementById(objId)
		}

		this.innerText = innerText

		this.fgcolor = fgcolor
		this.rotation = rotateDeg
		this.posLeft = posLeft
		this.posTop = posTop

		this.object.className = classes
	}

	set fgcolor(newColor) {
		this.object.style.color = newColor
	}

	set bgcolor(newColor) {
		this.object.style.backgroundColor = newColor
	}

	set rotation(newRotation) {
		this.object.style.transform = "rotate(" + newRotation + "deg)"
	}

	set posLeft(newPos) { 
		this.object.style.left = newPos + "%"
	}

	set posTop(newPos) {
		this.object.style.top = newPos + "%"
	}

	set innerText(text) {
		this.object.innerText = text
	}

	setParent(parentId) {
		document.getElementById(parentId).appendChild(this.object)
	}
}

class Clock extends Object {
	constructor() {
		super("time", 0, "HELLO", "#ffffff")
	}

	updateTime() {		
		let date = new Date();
		
		let time = [date.getHours(), date.getMinutes(), date.getSeconds()]
		let stringTime = ""

		for (let i=0; i < time.length; i++) {
			if (time[i] < 10) {
				stringTime += "0"
			}

			stringTime += time[i]

			if (i < time.length-1){
				stringTime += ":"
			}
		}
	
		document.getElementById("time").innerText = stringTime
	}

	changeColor(){
		this.bgcolor = RandColor();
		this.fgcolor = RandColor();
	}
}

const clock = new Clock()
const tickSound = new Audio("tick.mp4")
const tockSound = new Audio("tock.mp4")

var randObjects = []

function ClearContainer(){
	var div = document.getElementById('random-container');
	div.innerHTML = "";
}

function randNumber(max){
	return Math.floor(Math.random() * max)
}

function RandColor(){
	red = randNumber(255);
	green = randNumber(255);
	blue = randNumber(255);
	
	return "rgb(" + red + ", " + green + ", " + blue + ")"
}

settings = [1000,1000]

window.setInterval(clock.updateTime, 1000); //Update clock interval

window.setInterval(function() { //Change clock colour interval
	clock.changeColor()
},settings[0])

window.setInterval(function() { //Add tick tock interval
	var word;

	if (randObjects.length == 0 || randObjects[randObjects.length-1].object.innerText == "tock"){
		word = "tick"
		tickSound.play()
	}
	else {
		word = "tock"
		tockSound.play()
	}

	randObjects.push(new Object(objId="none", rotateDeg=randNumber(360), innerText=word, fgcolor=RandColor(), bgcolor="#ffffff", classes='random-word', posLeft=randNumber(100), posTop=randNumber(100)))
	randObjects[randObjects.length-1].setParent('random-container')

	if (randObjects.length > 500) {
		randObjects.shift()
	}
}, settings[1]);