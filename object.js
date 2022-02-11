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
