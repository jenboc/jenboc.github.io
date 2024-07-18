// BASIC GRAPHICS CODE 

const header = document.getElementById("header");

const lerp = (a, b, t) => a + t * (b - a);
const randfloat = (min, max) => min + Math.random() * (max - min + 1);

class Colour 
{
    constructor(r, g, b, a) 
    {
        this.r = r;
        this.g = g;
        this.b = b;
        this.a = a;
    }

    toString() 
    {
        return `rgba(${this.r}, ${this.g}, ${this.b}, ${this.a})`;
    }

    static lerp(a, b, t) 
    {
        return new Colour(lerp(a.r, b.r, t), lerp(a.g, b.g, t), lerp(a.b, b.b, t), lerp(a.a, b.a, t));
    }

    static random()
    {
        return new Colour(randfloat(0, 255), randfloat(0, 255), randfloat(0, 255), randfloat(0, 1))
    }
}

class Vec2
{
    constructor(x, y)
    {
        this.x = x;
        this.y = y;
    }

    static add(a, b)
    {
        return new Vec2(a.x + b.x, a.y + b.y);
    }

    static subtract(a, b) 
    {
        return new Vec2(a.x - b.x, a.y - b.y);
    }

    static multiplyScalar(a, b)
    {
        if (typeof(a) == "number")
        {
            return new Vec2(b.x * a, b.y * a);
        }
        else 
        {
            return new Vec2(a.x * b, a.y * b);
        }
    }

    static lerp(a, b, t) 
    {
        return new Vec2(lerp(a.x, b.x, t), lerp(a.y, b.y, t));
    }

    static zero() 
    {
        return new Vec2(0, 0);
    }

    static fromAngle(radians) 
    {
        return new Vec2(Math.cos(radians), Math.sin(radians));
    }
    
    toString() 
    {
        return `(${this.x}, ${this.y})`;
    }
}

class Canvas 
{
    #dom;
    #ctx;
    #backgroundColour;
    #sprites; 
    #spawners;

    #tickComplete;

    constructor(canvasId, backgroundColour) 
    {
        this.#dom = document.getElementById(canvasId);
        this.#ctx = this.#dom.getContext("2d");
        this.#backgroundColour = backgroundColour;
        this.#sprites = []
        this.#spawners = []

        this.#tickComplete = true;
    }

    addSprite(sprite) 
    {
        this.#sprites.push(sprite);
    }

    addSpawner(spawner) 
    {
        this.#spawners.push(spawner);
    }

    updateDimensions() 
    {
		const rect = header.getBoundingClientRect();
        this.#dom.height = Math.ceil(rect.top + rect.height);
        this.#dom.width = Math.ceil(rect.width);
    }

    update() 
    {
        console.log("update");
        this.#spawners.forEach(s => s.update());

        const toDelete = [];

        for (let i = 0; i < this.#sprites.length; i++)
        {
            const sprite = this.#sprites[i];
            sprite.update();
            
            if (sprite.delete)
                toDelete.push(i);
        }

        toDelete.forEach(index => this.#sprites.splice(index, 1));
    }

    draw() 
    {
        console.log("draw");
        this.#sprites.forEach(sprite => sprite.draw(this.#ctx));
    }

    clear() 
    {
        this.#ctx.fillStyle = this.#backgroundColour.toString();
        this.#ctx.fillRect(0, 0, this.#dom.width, this.#dom.height)
    }

    tick(timestamp) 
    {
        console.log("tick");
        this.clear();
        this.update();
        this.draw();
        console.log("end tick");
		requestAnimationFrame(() => this.tick());
    }

	start()
	{
		requestAnimationFrame(() => this.tick());
	}

    getWidth() 
    {
        return this.#dom.width;
    }

    getHeight() 
    {
        return this.#dom.height;
    }
}

class Sprite
{
    _position;
    _velocity; 
    _acceleration; 

    constructor(position, velocity, acceleration)
    {
        this._position = position;
        this._velocity = velocity;
        this._acceleration = acceleration;
        this.delete = false;
    }

    update() 
    {
        this._velocity = Vec2.add(this._velocity, this._acceleration);
        this._position = Vec2.add(this._position, this._velocity); 
    }

    draw(ctx) 
    {
        throw new Error("draw() is not implemented");
    }
}

class ParticleSettings
{
    constructor(startColour, endColour, startRadius, endRadius, lifetime)
    {
        this.startColour = startColour;
        this.endColour = endColour;
        this.startRadius = startRadius;
        this.endRadius = endRadius;
        this.lifetime = lifetime;
    }

    getColour(timeAlive) 
    {
        const t = timeAlive / this.lifetime;
        return Colour.lerp(this.startColour, this.endColour, t);
    }

    getRadius(timeAlive) 
    {
        const t = timeAlive / this.lifetime;
        return lerp(this.startRadius, this.endRadius, t);
    }

    isAlive(timeAlive) 
    {
        return timeAlive < this.lifetime;
    }
}

class Particle extends Sprite 
{
    #settings;
    #timeAlive;

    constructor(position, velocity, gravity, settings) 
    {
        super(position, velocity, new Vec2(0, gravity));
        
        this.#settings = settings;
        this.#timeAlive = 0;
    }

    update() 
    {
        this.#timeAlive++;

        super.update();

        this.delete = !this.#settings.isAlive(this.#timeAlive);
    }

    draw(ctx) 
    {
        const radius = this.#settings.getRadius(this.#timeAlive);

        if (radius <= 0) return;

        ctx.fillStyle = this.#settings.getColour(this.#timeAlive).toString(); 
        ctx.beginPath(); 
        ctx.arc(this._position.x, this._position.y, radius, 0, 2 * Math.PI);
        ctx.fill();
    }
}
