words = ["tick", "tock"]

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
		super("time", 0, "HELLO")
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

function ClearContainer(){
	var div = document.getElementById('random-container');
	div.innerHTML = "";
	localStorage.ticktock = div.innerHTML
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


function AddRandom(){
	var newTickTock = new Object(objId="none", rotateDeg=randNumber(360), innerText=words[randNumber(words.length)], fgcolor=RandColor(), bgcolor="#ffffff", classes='random-word', posLeft=randNumber(100), posTop=randNumber(100))
	newTickTock.setParent('random-container')
}

function cchangecolor(){
	clock.changeColor()
}

function GetSettings() {
	var query = window.location.search;
	var parameters = new URLSearchParams(query);

	var settings = [1000, 1000]

	if (parameters.has('cchange')){
		cchange = parameters.get('cchange')
	}

	if (parameters.has('tadd')){
		tadd = parameters.get('tadd')
	}

	return settings
}

settings = GetSettings()

window.setInterval(clock.updateTime, 1000);
window.setInterval(cchangecolor,settings[0])
window.setInterval(AddRandom, settings[1]);