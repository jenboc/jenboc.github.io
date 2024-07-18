// CODE FOR THE FIREWORKS

const FIREWORK_COLOUR = new Colour(255, 255, 255, 1);
const FIREWORK_RADIUS = 3;

class Firework extends Sprite 
{
    #canvas
    #lifetime 
    #timeAlive
    #numSparks 

    constructor(canvas, position, velocity, gravity, lifetime, numSparks) 
    {
        super(position, velocity, new Vec2(0, gravity));

        this.#canvas = canvas;
        this.#lifetime = lifetime;
        this.#timeAlive = 0;
        this.#numSparks = numSparks;
    }

    update() 
    {
        super.update(); 

        this.#timeAlive++;
        if (this.#timeAlive < this.#lifetime || this.delete) return; 
        this.delete = true; 
           
        const startColour = Colour.random();
        startColour.a = 1;
        const endColour = Colour.random();
        endColour.a = 0;

        const startRadius = 3;
        const endRadius = 1;
        
        const settings = new ParticleSettings(startColour, endColour, startRadius, endRadius, this.#lifetime * 5);

        for (let i = 1; i < this.#numSparks; i++)
        {
            const velocity = Vec2.multiplyScalar(Vec2.fromAngle(randfloat(0, 90)), randfloat(1, 3));

            const spark = new Particle(this._position, velocity, this._acceleration.y, settings);

            this.#canvas.addSprite(spark)
        }
    }

    draw(ctx) 
    {
        ctx.fillStyle = FIREWORK_COLOUR.toString(); 
        
        ctx.beginPath(); 
        ctx.arc(this._position.x, this._position.y, FIREWORK_RADIUS, 0, 2 * Math.PI); 
        ctx.fill();
    }
}

class FireworkLauncher 
{
    #canvas;
    
    #minDelay;
    #maxDelay;
    #currentDelay;
    #maxAngle; 
    #gravity; 
    #minLifetime;
    #maxLifetime;
    #minSpeed;
    #maxSpeed;

    #minSparks;
    #maxSparks;

    constructor(canvas, minSpawnDelay, maxSpawnDelay, maxAngle, gravity, maxLifetime, minLifetime, minSpeed, maxSpeed, minSparks, maxSparks) 
    {
        this.#canvas = canvas;
        this.#minDelay = minSpawnDelay;
        this.#maxDelay = maxSpawnDelay;
        this.#currentDelay = randfloat(this.#minDelay, this.#maxDelay);
        this.#maxAngle = maxAngle;
        this.#gravity = gravity;

        // Swap min and max spawn height since top = 0
        this.#minLifetime = minLifetime;
        this.#maxLifetime = maxLifetime;

        this.#minSpeed = minSpeed;
        this.#maxSpeed = maxSpeed;
        
        this.#minSparks = minSparks;
        this.#maxSparks = maxSparks;
    }

    update()
    {
        this.#currentDelay--; 
        if (this.#currentDelay > 0) return; 

        const position = new Vec2(randfloat(0, this.#canvas.getWidth()), this.#canvas.getHeight());
        const velocity = Vec2.multiplyScalar(randfloat(this.#minSpeed, this.#maxSpeed), Vec2.fromAngle(randfloat(-this.#maxAngle, this.#maxAngle) - Math.PI / 2));

        const lifetime = randfloat(this.#minLifetime, this.#maxLifetime); 

        const numSparks = randfloat(this.#minSparks, this.#maxSparks);

        const firework = new Firework(this.#canvas, position, velocity, this.#gravity, lifetime, numSparks);
        this.#canvas.addSprite(firework);

        this.#currentDelay = randfloat(this.#minDelay, this.#maxDelay);
    }
}