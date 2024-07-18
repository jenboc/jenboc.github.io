// Adds an event to all typewriter objects to remove the blinking cursor once the typing has finished

document.querySelectorAll(".typewriter").forEach((element) => {
	element.addEventListener("animationend", (e) => {
		if (e.animationName === "typing-anim")
		{
			// We can remove the typing animation as well since it has ended
			e.target.style.animation = "none";
			e.target.style.border = "none";
		}
	});
});
