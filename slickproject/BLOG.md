# Moving a Wizard around a Slick2D Tile Map

## Why this experiment exists

This experiment builds a minimal Slick2D game loop around a tile map, animated
wizard sprite, keyboard movement, and collision checks based on TileD map
properties. It is a compact example of the basic 2D game structure: initialize
assets, update state from input and elapsed time, render the map and sprite,
and prevent movement into blocked tiles.

For experienced Java developers, the interesting part is the boundary between
general application code and game-loop code. The `update` method is called
repeatedly with a `delta` value. Movement uses that delta instead of fixed
pixels per frame. Animation advances only when movement occurs. Collision is
not a physics engine; it is a lookup into a precomputed boolean grid derived
from the tile map.

The module is also useful because it shows the runtime assumptions of old
desktop Java game stacks. The code compiles with the Slick2D dependency, but
running it requires graphical support and the referenced image assets to be
available under the expected resource paths.

## Execution path

The entrypoint creates an `AppGameContainer`, sets a window size, and starts the
Slick2D loop:

```java
AppGameContainer app = new AppGameContainer(new WizardGame());
app.setDisplayMode(500, 400, false);
app.start();
```

`WizardGame` extends `BasicGame`, so Slick calls `init`, then repeatedly calls
`update` and `render`. The constructor only names the game window with
`super("Wizard game")`. All real setup happens in `init`, which matters because
Slick resources such as `Image` and `TiledMap` are loaded in the context of the
game container, not as static application initialization. The class also keeps
the current position as `private float x = 34f, y = 34f`, so updates can move by
fractional amounts even though rendering casts coordinates to integers.

```java
// Float position allows delta-based movement between integer draw calls.
private float x = 34f, y = 34f;
```

The starting coordinate aligns with the 34-pixel tile size, so the wizard begins
one tile in from the origin.

## Core code walkthrough

The `init` method loads two frames for each direction:

```java
Image[] movementUp = {new Image("data/wmg1_bk1.png"), new Image("data/wmg1_bk2.png")};
Image[] movementDown = {new Image("data/wmg1_fr1.png"), new Image("data/wmg1_fr2.png")};
Image[] movementLeft = {new Image("data/wmg1_lf1.png"), new Image("data/wmg1_lf2.png")};
Image[] movementRight = {new Image("data/wmg1_rt1.png"), new Image("data/wmg1_rt2.png")};
int[] duration = {300, 300};
grassMap = new TiledMap("data/grassmap.tmx");
```

Each direction gets its own animation. The map is a TileD `.tmx` file stored as
a resource. The tile size is 34 pixels, which is also the sprite collision size
used later.

Animations are created with auto-update disabled by passing `false` to the
`Animation` constructor. The initial `sprite` is `right`. Because auto-update is
disabled, frames advance only when the code calls `sprite.update(delta)`.
Standing still keeps the current frame.

```java
// false means animation advances only when movement code calls update(delta).
up = new Animation(movementUp, duration, false);
down = new Animation(movementDown, duration, false);
left = new Animation(movementLeft, duration, false);
right = new Animation(movementRight, duration, false);
```

That makes the animation state a direct consequence of input handling.

Collision data is extracted from tile properties:

```java
blocked = new boolean[grassMap.getWidth()][grassMap.getHeight()];

for (int xAxis = 0; xAxis < grassMap.getWidth(); xAxis++) {
    for (int yAxis = 0; yAxis < grassMap.getHeight(); yAxis++) {
        int tileID = grassMap.getTileId(xAxis, yAxis, 0);
        String value = grassMap.getTileProperty(tileID, "blocked", "false");
```

The `.tmx` file marks rock tiles with `blocked=true`. Precomputing that into a
boolean array avoids property lookups on every movement check.

The map file itself stores tile data as base64 gzip, which Slick's `TiledMap`
loader decodes. The Java code does not parse that encoding directly; it trusts
the library to expose width, height, tile ids, and tile properties. That is the
right level of abstraction for the experiment because movement code should not
know about the serialized map format.

## Important implementation details

Movement is based on keyboard state and elapsed time:

```java
if (input.isKeyDown(Input.KEY_UP)) {
    sprite = up;
    if (!isBlocked(x, y - delta * 0.1f)) {
        sprite.update(delta);
        y -= delta * 0.1f;
    }
}
```

The same pattern is repeated for down, left, and right. The collision check
looks ahead to the proposed edge of the sprite. Down and right add `SIZE` so
the far edge is tested rather than the top-left corner.

Rendering is deliberately simple: `grassMap.render(0, 0)` draws the map, then
`sprite.draw((int) x, (int) y)` draws the current animation over it. No camera,
scaling, layers, or UI overlay are involved.

```java
// Draw world first, then draw the actor at its current screen position.
grassMap.render(0, 0);
sprite.draw((int) x, (int) y);
```

This is the entire render order, which is why there are no z-index or camera
concerns in the current experiment.

The collision helper converts world coordinates to tile coordinates:

```java
private boolean isBlocked(float x, float y) {
    int xBlock = (int) x / SIZE;
    int yBlock = (int) y / SIZE;
    return blocked[xBlock][yBlock];
}
```

That assumes coordinates remain inside the map bounds. The current movement
code does not clamp or guard array access.

The `SIZE` constant couples sprite dimensions, tile dimensions, and collision
math. That works because the map uses 34-by-34 tiles and the wizard sprite is
treated as a 34-pixel square. If either asset changes size, collision behavior
would need to be revisited or driven from map and image metadata instead of a
single constant.

## Runtime behavior and caveats

The game window is 500 by 400 pixels. The map in `grassmap.tmx` is 10 by 10
tiles at 34 pixels each, so the map itself is 340 by 340 pixels. The player
starts at `(34, 34)` facing right and moves with arrow keys.

The biggest runtime caveat is assets. The repository includes `grassmap.tmx`,
but the Java source references PNG files such as `data/wmg1_bk1.png` and a
tileset image such as `grass.png`. If those resources are absent from the
runtime classpath, Slick will fail during initialization. The module compiles
without those assets because resource existence is checked at runtime.

The collision model is also intentionally simple. It checks one point per
movement direction, so corners and partial overlap are not handled robustly.
Out-of-bounds movement can index outside `blocked`. The `delta * 0.1f` speed
can also skip over narrow obstacles if frame times are large. This is a
movement experiment, not a complete tile collision engine.

## Suggested next experiments

Add bounds checks to `isBlocked` and decide whether outside-map movement should
be blocked or clamped. Check both relevant sprite corners for each movement
direction. Add missing runtime assets or replace them with generated placeholder
resources so the demo launches from a clean checkout. Introduce a camera and
separate world coordinates from screen coordinates. Finally, write a small test
around the tile-property collision map so map edits do not silently change
walkability.

A natural gameplay follow-up is diagonal movement. Adding it would require
normalizing speed so diagonal travel is not faster than horizontal or vertical
travel, and it would make collision checks exercise two axes in the same frame.
That change would quickly show whether the current point-sampling collision
model is good enough.
It would also make animation priority decisions visible. That matters once two
movement keys are held together.

A second follow-up would load movement speed, tile size, and starting position
from configuration or map metadata. That would remove several hidden constants
from the code and make asset changes less likely to break movement.
It would also make maps with different tile sizes practical and easier to tune
while playtesting.
