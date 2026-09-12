# Realistic Video Generation Prompts

## Tool used

Google gemini 3.8-flash (extended)

## Global realism instruction

Add this to **every RGB prompt**:

```text
IMPORTANT REALISM REQUIREMENT:

This must look like an actual operational field camera recording, not a stock video, cinematic film, commercial, game render, or AI-generated landscape.

Use realistic camera imperfections:
- slight stabilization micro-jitter
- subtle mechanical vibration
- imperfect exposure adaptation
- realistic motion blur during camera movement
- realistic rolling-shutter or electronic-shutter characteristics
- realistic digital sharpening
- mild compression artifacts
- limited dynamic range
- slight sensor noise
- imperfect white balance
- realistic atmospheric haze
- uneven illumination
- natural shadows
- realistic lens characteristics
- occasional small focus and exposure adjustments

The camera movement must feel practical and observational rather than cinematic.

Do not use cinematic camera movements, dramatic composition, artificial depth of field, dramatic color grading, slow motion, perfect sharpness, or an unrealistically clean image.

The scene should contain ordinary environmental clutter and randomness. Do not arrange objects neatly for the camera.

When a person is present, they should occupy a realistic and relatively small portion of the frame rather than being centered and presented as a hero subject.

The footage should look slightly imperfect and ordinary, like real operational footage captured during an active search.
```

---

# Desert — Positive

```text
Create approximately 10 seconds of photorealistic operational field footage showing a remote desert search area.

The camera is positioned at an elevated viewpoint and looks approximately 30–45 degrees downward toward the ground. It is moving slowly forward while scanning the terrain.

A SINGLE real adult person is somewhere on the ground, approximately 30–70 meters from the camera.

The person is NOT posing for the camera. They are walking naturally across the terrain and occasionally partially blend into the surroundings.

The person should be relatively small in the frame, roughly 1–4% of the image area depending on distance.

Use an ordinary natural desert environment rather than a spectacular landscape:
irregular sand, compacted soil, small rocks, tire tracks, scattered scrub vegetation, dry bushes, small depressions, footprints, stones, and subtle changes in terrain.

The person should interact naturally with the terrain. Their feet must contact the ground correctly and their shadow must have physically plausible direction and softness.

Do not deliberately track or center the person. The person should remain somewhere off-center as would happen during a genuine search sweep.

Lighting should resemble harsh natural daytime sunlight with realistic exposure. Do not make the image excessively saturated or orange.

Include realistic atmospheric dust and haze toward the distance.

Camera movement should have subtle stabilization imperfections and physically plausible motion.

Do not make the person unusually large, perfectly visible, brightly colored, centered, or sharply isolated.

No cinematic composition.
No dramatic camera movement.
No artificial objects.
No text.
No HUD.
No bounding boxes.
No watermarks.
```

---

# Desert — Hard Negative

```text
Create approximately 10 seconds of photorealistic operational field footage showing an empty remote desert.

THERE MUST BE NO HUMAN BEING IN THIS VIDEO.

The scene should contain a naturally occurring visual pattern that could plausibly produce a false positive in a person detector.

Use one ambiguous object such as a discarded dark jacket or piece of fabric partially caught against a low bush, combined with nearby rocks and shadows.

It must NOT look deliberately arranged as a human mannequin or fake person.

From the camera's viewpoint, the object should briefly resemble the rough appearance of a person at a distance, but closer visual inspection should reveal that it is only fabric, vegetation, rocks, and shadow.

The object should be relatively small in the frame and located off-center.

The surrounding desert should contain realistic clutter:
rocks, scrub, tire tracks, footprints, irregular sand, dry vegetation, shadows, small depressions, and scattered debris.

The camera is slowly moving forward during a normal search sweep. It should NOT move toward the ambiguous object specifically.

Use realistic camera characteristics:
subtle stabilization movement, vibration, motion blur, compression, sensor noise, atmospheric haze, imperfect exposure, and natural lighting.

Do not create a recognizable human silhouette.
Do not create a mannequin.
Do not create a statue.
Do not create a person wearing clothing.
Do not add any human body parts.

The scene must remain continuous for the entire clip.

No cinematic composition.
No dramatic lighting.
No text.
No HUD.
No bounding boxes.
No watermarks.
```

---

# Desert — Clear Negative

```text
Create approximately 10 seconds of photorealistic operational field footage over a completely empty desert search area.

NO PEOPLE.
NO HUMAN-LIKE OBJECTS.
NO CLOTHING.
NO BACKPACKS.
NO MANNEQUINS.
NO STATUES.
NO INTENTIONAL HUMAN-SHAPED DECOYS.

The camera is performing a routine search sweep and slowly moving forward.

Show ordinary, visually messy desert terrain:
compact sand, gravel, rocks of different sizes, dry scrub, tire tracks, footprints from previous activity, small depressions, scattered natural debris, and irregular terrain.

Do not make the landscape beautiful or cinematic. It should look like an ordinary area being searched.

Use realistic sunlight, shadows, atmospheric haze, camera vibration, stabilization imperfections, compression, sensor noise, exposure variation, and natural motion blur.

The camera should occasionally reveal more terrain as it moves forward, but the environment must remain the same continuous location.

No person should appear at any point, including in the far background.

No cinematic camera movement.
No dramatic color grading.
No text.
No HUD.
No bounding boxes.
No watermarks.
```

---

# Forest — Positive

```text
Create approximately 10 seconds of photorealistic operational field footage showing a remote forest search area.

A SINGLE real adult person is present on the ground.

The person is approximately 20–50 meters from the camera and is partially obscured by natural vegetation.

They are walking slowly through a small irregular clearing between trees rather than standing in an open field.

The camera is moving slowly forward and slightly sideways as part of a practical search sweep.

The person should NOT look toward the camera, pose, wave, or behave like an actor.

Their clothing should be ordinary outdoor clothing with muted, realistic colors.

The person should occupy only a small part of the frame and should sometimes be partially obscured by branches or leaves.

The forest must be dense and visually messy:
different tree sizes, branches, leaves, undergrowth, fallen branches, exposed soil, rocks, dead vegetation, patches of sunlight, and irregular shadows.

Use realistic depth and atmospheric perspective.

Some branches should pass between the camera and the person. Do not artificially outline or separate the person from the background.

Use realistic stabilization imperfections, micro-jitter, motion blur, compression, sensor noise, exposure adaptation, and occasional loss of fine detail.

This should look like an actual operational search recording rather than a cinematic forest scene.

No dramatic camera movement.
No perfect lighting.
No centered subject.
No artificial depth of field.
No text.
No HUD.
No bounding boxes.
No watermark.
```

---

# Forest — Hard Negative

```text
Create approximately 10 seconds of photorealistic operational field footage over dense forest.

THERE MUST BE NO HUMAN BEING IN THE VIDEO.

Create a naturally occurring hard negative that could plausibly confuse a person detector.

For example, a fallen branch and dark piece of fabric partially trapped among vegetation should create an irregular shape that, from the camera's distance, vaguely resembles a crouching or standing person.

The object must be naturally integrated into the environment.

It must NOT be a mannequin or intentionally constructed human dummy.

Surround it with ordinary forest clutter:
branches, leaves, tree trunks, rocks, exposed soil, dead vegetation, and shadows.

The ambiguous pattern should be relatively small in the frame and partially occluded.

The camera should continue a normal search sweep rather than deliberately filming the object.

Use realistic camera footage characteristics:
small stabilization movements, vibration, motion blur, sensor noise, compression artifacts, exposure changes, and imperfect stabilization.

Do not make the ambiguous object obviously human.

NO real person.
NO mannequin.
NO statue.
NO human body parts.
NO face.
NO cinematic staging.

No text.
No HUD.
No bounding boxes.
No watermark.
```

---

# Forest — Clear Negative

```text
Create approximately 10 seconds of photorealistic operational field footage over an empty forest.

There is ABSOLUTELY NO PERSON anywhere in the scene.

There are also no clothing items, backpacks, mannequins, statues, human-shaped objects, or intentionally ambiguous human decoys.

The camera is performing a routine search sweep over ordinary dense woodland.

Show naturally occurring visual clutter:
tree trunks, branches, leaves, bushes, fallen logs, rocks, exposed soil, patches of sunlight, deep shadows, and irregular terrain.

The composition should be imperfect and observational. Do not make it look like a nature documentary.

The camera slowly moves forward with realistic stabilization and minor vibration.

Include realistic compression, sensor noise, motion blur, exposure adaptation, and changing shadows as the camera moves.

Do not reveal a person later in the background.

No cinematic camera movement.
No dramatic lighting.
No artificial color grading.
No text.
No HUD.
No bounding boxes.
No watermark.
```

---

# Thermal Conversion Prompt

Use this in a **fresh conversation with the generated RGB video uploaded**:

```text
The uploaded video is the source recording for a multimodal dataset.

Transform this EXACT video into realistic thermal sensor footage.

This is a SENSOR CONVERSION task, NOT a video regeneration task.

Preserve EXACTLY:

- frame count
- frame ordering
- duration
- camera trajectory
- camera orientation
- camera movement
- scene geometry
- every object's position
- every object's movement
- person's position and movement
- terrain
- vegetation
- shadows and their physical locations
- occlusions
- background
- timing

Do not invent new events.

Do not remove existing objects.

Do not add people.

Do not move objects.

Do not change the camera viewpoint.

Do not zoom.

Do not crop.

Do not alter the framing.

Do not stabilize the footage differently.

Do not change the scene.

The output must be temporally aligned with the source RGB video frame-for-frame.

The output must represent the SAME recording captured using a thermal sensor.

For THERMAL:

Render physically plausible thermal intensity based on apparent material temperature.

Living people should have plausible thermal signatures.

Vegetation, rocks, ground, fabric, vehicles, and other materials should exhibit different thermal responses according to realistic thermal behavior.

Thermal appearance should depend on the scene and materials rather than simply applying a grayscale or color filter.

Preserve all existing motion and occlusions exactly.

Do not create or remove details simply to make the thermal image look better.

Include realistic thermal sensor limitations, noise, resolution characteristics, contrast limitations, and occasional sensor artifacts.

The result should look like a mediocre real thermal recording rather than a perfect synthetic thermal image.

No text.
No labels.
No bounding boxes.
No HUD.
No watermark.
```

---

# NIR Conversion Prompt

Use this in a **fresh conversation with the generated RGB video uploaded**:

```text
The uploaded video is the source recording for a multimodal dataset.

Transform this EXACT video into realistic near-infrared (NIR) sensor footage.

This is a SENSOR CONVERSION task, NOT a video regeneration task.

Preserve EXACTLY:

- frame count
- frame ordering
- duration
- camera trajectory
- camera orientation
- camera movement
- scene geometry
- every object's position
- every object's movement
- person's position and movement
- terrain
- vegetation
- shadows and their physical locations
- occlusions
- background
- timing

Do not invent new events.

Do not remove existing objects.

Do not add people.

Do not move objects.

Do not change the camera viewpoint.

Do not zoom.

Do not crop.

Do not alter the framing.

Do not stabilize the footage differently.

Do not change the scene.

The output must be temporally aligned with the source RGB video frame-for-frame.

The output must represent the SAME recording captured using a near-infrared sensor.

For NIR:

Render realistic near-infrared sensor imagery based on plausible material reflectance.

Preserve the same geometry, objects, movement, camera trajectory, framing, and timing.

Produce plausible NIR reflectance differences between vegetation, skin, clothing, soil, rocks, sand, snow, and other materials.

Vegetation should have realistic NIR response.

Different materials should not all appear identical.

Do not simply convert the RGB footage to grayscale.

Do not apply a generic monochrome filter.

Do not invent additional texture or detail that was not present in the source.

Include realistic NIR sensor limitations, noise, contrast characteristics, exposure behavior, and image artifacts.

The result should look like a mediocre real NIR recording rather than a perfect synthetic image.

No text.
No labels.
No bounding boxes.
No HUD.
No watermark.
```

## Critical consistency rule

The most important requirement for the Thermal/NIR conversions is:

```text
DO NOT REGENERATE THE VIDEO.

TRANSFORM THE EXACT SOURCE VIDEO WHILE PRESERVING ITS GEOMETRY AND TEMPORAL STRUCTURE FRAME-FOR-FRAME.
```

The target is **realistic imperfect sensor footage**, not visually impressive footage. This matters because the specification assumes the RGB, Thermal, and NIR versions retain matching geometry for annotation propagation.


The uploaded video is the source recording for a computer-vision dataset.
Transform this EXACT video into realistic thermal infrared camera footage.
THIS IS A SENSOR-DOMAIN CONVERSION TASK, NOT A VIDEO REGENERATION, SCENE EDIT, OR ARTISTIC RESTYLING TASK.
The output must represent the SAME recording captured by a real thermal infrared camera.
The RGB video is the authoritative reference for scene geometry, motion, timing, and object identity.

1. EXACT SOURCE PRESERVATION
Preserve exactly:

frame count
frame ordering
duration
timing
camera trajectory
camera position
camera orientation
field of view
perspective
framing
camera movement
terrain geometry
vegetation placement
rock placement
object positions
object movement
person's position
person's movement
person's pose
object boundaries
occlusions
relative depth relationships
Every thermal frame must correspond to the same frame in the RGB source.
DO NOT:

add objects
remove objects
duplicate objects
move objects
change object shapes
add people
remove people
change the person's pose
change the person's movement
change the person's location
alter existing occlusions
reveal hidden objects
hide visible objects
zoom
crop
rotate
reframe
change the camera viewpoint
change the camera trajectory
stabilize the footage differently
create new events
regenerate the environment
The person's location and geometry must remain aligned with the source video.

2. THERMAL SENSOR CONVERSION
Convert the visual appearance into plausible monochrome thermal infrared imagery.
The output must NOT look like:

RGB converted to grayscale
RGB with increased contrast
an RGB image with a thermal color palette
a simple brightness inversion
a fixed semantic recoloring
Do not directly map RGB brightness to thermal brightness.
Do not assume that bright RGB regions are thermally hot or that dark RGB regions are thermally cold.
Instead, represent plausible apparent-temperature differences between the EXISTING surfaces and objects in the scene.
Use realistic thermal differences between:

human body
clothing
exposed skin
rocks
sand
compacted soil
vegetation
dry bushes
shaded terrain
sun-exposed terrain
Use a natural monochrome thermal presentation with realistic grayscale intensity variation.

3. NATURAL THERMAL VARIATION
Do NOT use a fixed rule such as:
person = white
rocks = white
sand = gray
vegetation = dark gray
Thermal intensity must vary naturally.
A person does NOT need to be the brightest object in the frame.
A sun-heated rock may be as bright as, or brighter than, parts of a person.
A shaded rock may be substantially darker.
Different rocks may have different thermal intensities.
Sand should contain natural spatial variation rather than being one uniform gray.
Vegetation should contain thermal variation rather than always being uniformly dark.
Different parts of a person may have different thermal intensity.
Thermal intensities of different materials may overlap.
Do not artificially increase human-versus-background contrast.
Do not artificially decrease human-versus-background contrast.
Allow the thermal appearance to emerge from plausible scene conditions.

4. DESERT THERMAL BEHAVIOR
For the existing desert environment, preserve every terrain feature and object from the source.
Represent plausible effects of:

direct solar heating
shade
material differences
thermal inertia
surface exposure
local environmental conditions
distance
atmospheric attenuation
Sun-exposed rocks and ground may appear thermally warm.
Shaded surfaces may have different apparent temperatures.
Do not make the entire desert uniformly hot.
Do not deliberately create thermal patterns resembling humans.
Do not deliberately create or remove false positives.
Any ambiguity must arise naturally from the existing scene.

5. HUMAN THERMAL SIGNATURE
If a person is present in the source:
Preserve their exact:

location
scale
pose
orientation
movement
occlusion
relationship to the terrain
Represent the person with a plausible thermal signature.
Do NOT:

make the person artificially glow
add a bright halo
make the person pure white by default
make the person automatically the brightest object
sharpen the person
enlarge the person
center the person
improve their visibility artificially
The person's thermal contrast should depend on the surrounding environment.
If the person is partially occluded in the RGB source, preserve that occlusion.
Do not hallucinate hidden body parts.

6. THERMAL SENSOR LIMITATIONS
The output should resemble footage from a real, relatively low-cost or mediocre thermal camera mounted on an operational UAV.
Include realistic limitations such as:

limited spatial resolution
reduced fine detail
sensor noise
mild temporal noise
imperfect contrast
limited dynamic range
subtle fixed-pattern noise
mild quantization
realistic motion blur
mild compression artifacts
atmospheric attenuation
reduced separation between thermally similar surfaces
Do not make the thermal imagery look like a laboratory-grade scientific visualization.
Do not make it perfectly clean.
Do not add excessive noise or artifacts that obscure the scene.
The footage should remain usable for object detection.

7. TEMPORAL CONSISTENCY
Maintain consistent thermal appearance throughout the entire video.
Do not introduce:

flickering objects
disappearing objects
appearing objects
randomly changing temperatures
unstable human signatures
inconsistent terrain
frame-to-frame geometry changes
Thermal intensity may change gradually when physically plausible, but must remain temporally coherent.
Preserve all source motion exactly.

8. SHADOWS AND OCCLUSIONS
Preserve the physical location and geometry of all shadows and occluding objects from the source.
However, do NOT simply reproduce RGB shadows as black shapes.
Their thermal appearance should change naturally according to plausible surface temperature.
The physical geometry of the scene must remain unchanged.

9. DATASET FIDELITY
This video will be used as training data for a person detector.
Therefore:

preserve object geometry above visual aesthetics
preserve small targets
preserve natural background clutter
preserve difficult viewing conditions
preserve occlusions
preserve realistic thermal ambiguity
do not make people easier to detect for presentation purposes
do not clean up the scene
do not simplify the background
The goal is realistic sensor-domain variation while preserving the exact underlying recording.

10. OUTPUT RESTRICTIONS
No text.
No labels.
No bounding boxes.
No HUD.
No symbols.
No timestamps.
No watermarks.
No cinematic effects.
No dramatic color grading.
No artistic styling.
No new objects.
No deleted objects.
No camera changes.
No scene changes.
FINAL REQUIREMENT:
The result must look like the SAME original recording captured using a real thermal infrared camera.
It must NOT look like an RGB video with a grayscale or thermal filter applied.


**THERMAL DESERT PROMPT ABOVE**

---
**FOREST THERMAL PROMPT BELOW**

The uploaded video is the source recording for a computer-vision dataset.

Transform this EXACT video into realistic thermal infrared camera footage.

THIS IS A SENSOR-DOMAIN CONVERSION TASK, NOT A VIDEO REGENERATION, SCENE EDIT, OR ARTISTIC RESTYLING TASK.

The output must represent the SAME recording captured by a real thermal infrared camera.

The source RGB video is the authoritative reference for scene geometry, motion, timing, and object identity.

## 1. EXACT SOURCE PRESERVATION

Preserve exactly:

* frame count
* frame ordering
* duration
* timing
* camera trajectory
* camera position
* camera orientation
* field of view
* perspective
* framing
* camera movement
* terrain geometry
* tree positions
* branch positions
* vegetation placement
* rock positions
* fallen branches and logs
* soil and ground features
* all other existing objects
* person's position
* person's movement
* person's pose
* object boundaries
* occlusions
* relative depth relationships

Every thermal frame must correspond directly to the same frame in the RGB source.

DO NOT:

* add objects
* remove objects
* duplicate objects
* move objects
* change object shapes
* add people
* remove people
* change the person's pose
* change the person's movement
* change the person's location
* alter existing occlusions
* reveal hidden objects
* hide visible objects
* zoom
* crop
* rotate
* reframe
* change the camera viewpoint
* change the camera trajectory
* stabilize the footage differently
* create new events
* regenerate the environment

The person's location and geometry must remain aligned with the source video.

## 2. THERMAL SENSOR CONVERSION

Convert the visual appearance into plausible monochrome thermal infrared imagery.

The output must NOT look like:

* RGB converted directly to grayscale
* RGB with increased contrast
* RGB with a thermal color palette
* a simple brightness inversion
* a fixed semantic recoloring

Do not directly map RGB brightness or RGB color to thermal brightness.

Represent plausible apparent-temperature differences between the EXISTING surfaces and objects in the scene.

Use realistic thermal differences between:

* human body
* exposed skin
* clothing
* tree trunks
* branches
* leaves
* bushes
* fallen vegetation
* rocks
* soil
* exposed ground
* shaded surfaces
* sun-exposed surfaces

Use natural monochrome thermal intensity rather than artificial colors.

## 3. NATURAL THERMAL VARIATION

DO NOT use a fixed rule such as:

person = white
trees = dark
vegetation = dark
ground = gray

Thermal intensity must vary naturally within and between materials.

A person does NOT need to be the brightest object.

A sun-exposed tree trunk may be relatively warm.

A shaded tree trunk may be cooler.

Different leaves, branches, rocks, and ground surfaces may have different apparent temperatures.

Sunlit and shaded regions should not automatically have the same thermal intensity.

Thermal intensities of different materials may overlap.

The same material may have different thermal intensity in different parts of the scene.

Do not artificially maximize human-versus-background contrast.

Do not artificially minimize human-versus-background contrast.

Do not deliberately create thermal patterns resembling humans.

Allow realistic thermal ambiguity between the person and environmental objects.

## 4. FOREST THERMAL BEHAVIOR

Preserve the exact existing forest environment.

Represent plausible thermal variation caused by:

* direct sunlight
* shade
* canopy coverage
* material properties
* moisture differences
* exposed versus sheltered surfaces
* thermal inertia
* local environmental conditions
* distance
* atmospheric effects

Sun-exposed vegetation and surfaces may have different thermal intensity from shaded vegetation and surfaces.

Tree trunks, branches, leaves, soil, rocks, and fallen vegetation should not all have identical thermal intensity.

Do not make all vegetation uniformly cold.

Do not make all vegetation uniformly dark.

Do not make the entire forest thermally flat.

Do not deliberately manufacture false positives.

Any ambiguity must arise naturally from the existing scene.

## 5. HUMAN THERMAL SIGNATURE

If a person is present in the source video:

Preserve their exact:

* location
* scale
* pose
* orientation
* movement
* occlusion
* relationship to surrounding vegetation

Represent the person with a plausible thermal signature.

Do NOT:

* make the person artificially glow
* add a bright halo
* make the person pure white by default
* make the person automatically the brightest object
* sharpen the person
* enlarge the person
* center the person
* improve their visibility artificially

The person's thermal contrast should depend on the surrounding environment.

If clothing covers the body, do not represent the entire person as exposed skin.

Allow clothing and exposed skin to have different thermal appearances.

If the person is partially hidden by leaves, branches, bushes, or trees in the source, preserve those occlusions exactly.

Do not hallucinate hidden body parts.

Do not make thermal imagery "see through" vegetation that physically blocks the person in the source.

## 6. THERMAL SENSOR LIMITATIONS

The output should resemble footage from a real, relatively low-cost or mediocre thermal camera mounted on an operational UAV.

Include realistic limitations such as:

* limited spatial resolution
* reduced fine detail
* thermal sensor noise
* mild temporal noise
* imperfect contrast
* limited dynamic range
* subtle fixed-pattern noise
* mild quantization
* realistic motion blur
* mild compression artifacts
* atmospheric attenuation
* reduced separation between thermally similar surfaces

The thermal imagery should contain less fine visual detail than the original RGB footage where appropriate.

Do not make it look like a high-end scientific thermal visualization.

Do not make it perfectly clean.

Do not add excessive noise or artifacts that obscure the scene.

The footage must remain usable for person detection.

## 7. TEMPORAL CONSISTENCY

Maintain consistent thermal appearance throughout the entire video.

Do not introduce:

* flickering objects
* disappearing objects
* appearing objects
* randomly changing temperatures
* unstable human signatures
* inconsistent vegetation
* changing tree geometry
* frame-to-frame scene changes

Thermal intensity may change gradually when physically plausible, but must remain temporally coherent.

Preserve all source motion exactly.

## 8. SHADOWS AND OCCLUSIONS

Preserve the physical location and geometry of all shadows and occluding objects from the source.

Do NOT simply reproduce RGB shadows as black shapes.

Their thermal appearance should change naturally according to plausible surface temperature.

Preserve all branches, leaves, trunks, and vegetation that physically block the person.

Do not remove vegetation merely to improve human visibility.

## 9. DATASET FIDELITY

This video will be used as training data for a person detector.

Therefore:

* preserve small targets
* preserve natural forest clutter
* preserve vegetation occlusion
* preserve difficult viewing conditions
* preserve realistic thermal ambiguity
* preserve object geometry
* do not make people easier to detect for presentation purposes
* do not clean up the background
* do not simplify vegetation
* do not artificially outline the person

The goal is realistic sensor-domain variation while preserving the exact underlying recording.

## 10. OUTPUT RESTRICTIONS

No text.

No labels.

No bounding boxes.

No HUD.

No symbols.

No timestamps.

No watermarks.

No cinematic effects.

No dramatic color grading.

No artistic styling.

No new objects.

No deleted objects.

No camera changes.

No scene changes.

FINAL REQUIREMENT:

The result must look like the SAME original forest recording captured using a real thermal infrared camera.

It must NOT look like an RGB forest video with a grayscale filter applied.

Preserve the exact source video while replacing its RGB appearance with plausible thermal infrared sensor appearance.
