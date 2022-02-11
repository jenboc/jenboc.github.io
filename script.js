words = ["tick", "tock"]

class Object {
	constructor(objId="none", rotateDeg=10, innerText="none", fgcolor="#000000", bgcolor="#ffffff", classes="", posLeft="none", posTop="none") {
		if (objId == "none") {
			this.object = document.createElement("div")
		}
		else {
			this.object = document.getElementById(objId)
		}

		if (innerText != "none") {
			this.object.innerHTML = innerText
		}

		this.fgcolor = fgcolor
		this.bgcolor = bgcolor
		this.rotation = rotateDeg
		this.posLeft = posLeft
		this.posTop = posTop

		this.object.className = classes

		this.updateStyle()
	}

	updateStyle() {
		this.object.style.cssText = "color: " + this.fgcolor + ";background-color: " + this.bgcolor + ";"
		
		if (this.rotation > 0) {
			this.object.style.cssText += "transform: rotate(" + this.rotation + "deg);"
		}

		if (this.posLeft != "none" && this.posTop != "none") {
			this.object.style.cssText += "top: " + this.posTop + "%; left: " + this.posLeft + "%;"
		}
	}

	setParent(parentId) {
		document.getElementById(parentId).appendChild(this.object)
	}
}

class Clock extends Object {
	constructor() {
		super("time", 0)
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
	
		div.innerText = stringTime
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
	var newTickTock = new Object(objId="none", rotateDeg=randNumber(360), innerText=words[randNumber(words.length)], fgcolor=RandColor(), bgcolor="rgb(255,255,255)", classes='random-word', posLeft=randNumber(100), posTop=randNumber(100))
	newTickTock.setParent('random-container')
	localStorage.ticktock = div.innerHTML
}

function ChangeClockColor(){
	bgColor = RandColor();
	fgColor = RandColor();
	div = document.getElementById('time')
	div.style.cssText = "background-color: " + bgColor + ";color: " + fgColor + ";"
}

container = document.getElementById('random-container')
container.innerHTML = localStorage.ticktock

var query = window.location.search;
var parameters = new URLSearchParams(query);

var cchange = 0;
if (parameters.has('cchange')){
	cchange = parameters.get('cchange')
}
else{
	cchange = 500
}

var tadd = 0;
if (parameters.has('tadd')){
	tadd = parameters.get('tadd')
}
else {
	tadd = 1000
}

window.setInterval(clock.updateTime, 1000);
window.setInterval(ChangeClockColor,cchange);
window.setInterval(AddRandom, tadd);