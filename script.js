words = ["tick", "tock"]

class Object {
	constructor(objId="none", rotateDeg=0, innerText="none", fgcolor="#000000", classes="", posLeft="none", posTop="none") {
		if (objId == "none") {
			this.object = document.createElement("div")
			this.object.onclick = this.delete
		}
		else {
			this.object = document.getElementById(objId)
		}

		if (innerText != "none") {
			this.object.innerHTML = innerText
		}

		this.fgcolor = fgcolor
		this.rotation = rotateDeg
		this.posLeft = posLeft
		this.posTop = posTop

		this.object.className = classes

		this.updateStyle()
	}

	updateStyle() {
		this.object.style.color = this.fgcolor
		this.object.style.backgroundColor = this.bgcolor
		
		if (this.rotation > 0) {
			this.object.style.transform += "rotate(" + this.rotation + "deg)"
		}

		if (this.posLeft != "none" && this.posTop != "none") {
			this.object.style.top = this.posTop + "%"
			this.object.style.left = this.posLeft + "%"
		}
	}

	setParent(parentId) {
		document.getElementById(parentId).appendChild(this.object)
	}

	delete(){
		this.object.remove()
	}
}

class Clock extends Object {
	constructor() {
		super("time")
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
		do {
			this.bgcolor = RandColor();
			this.fgcolor = RandColor();
		} while (this.bgcolor == this.fgcolor)

		this.updateStyle()
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
	var newTickTock = new Object(objId="none", rotateDeg=randNumber(360), innerText=words[randNumber(words.length)], fgcolor=RandColor(), classes='random-word', posLeft=randNumber(100), posTop=randNumber(100))
	newTickTock.setParent('random-container')
	localStorage.ticktock = document.getElementById('random-container')
}


function cchangecolor(){
	clock.changeColor()
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
window.setInterval(cchangecolor,cchange)
window.setInterval(AddRandom, tadd);