# Realistic Video Generation Prompts

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

# Snow — Positive

```text
Create approximately 10 seconds of photorealistic operational field footage showing a remote snowy search area.

A SINGLE real adult person is present on the ground.

The person is approximately 30–80 meters from the camera and is relatively small in the image.

They are walking naturally across uneven snow rather than standing in a staged pose.

Their clothing should be realistic winter clothing with muted colors and physically plausible interaction with the environment.

Show footprints behind the person where appropriate.

Their feet must correctly contact the snow and their shadow must match the lighting.

The snow surface should NOT be perfectly smooth.

Include wind-formed patterns, compacted areas, footprints, exposed rocks, small vegetation, uneven depth, and subtle changes in snow texture.

The person should be somewhat difficult to detect at first glance because of distance, lighting, and environmental similarity, while still being genuinely visible.

Use realistic overcast or weak winter sunlight rather than dramatic golden-hour lighting.

The camera should move slowly during a practical search sweep with slight stabilization imperfections.

Include realistic exposure, motion blur, sensor noise, compression, and atmospheric conditions.

Do not center the person.
Do not make them oversized.
Do not make the snow perfectly white.
Do not use cinematic composition.

No text.
No HUD.
No bounding boxes.
No watermark.
```

---

# Snow — Hard Negative

```text
Create approximately 10 seconds of photorealistic operational field footage over a remote snowy area.

THERE MUST BE NO HUMAN BEING PRESENT.

Include one naturally occurring hard negative that could plausibly cause a false person detection.

For example, use a dark backpack partially covered by snow next to a low rock and a shadow, creating an irregular shape that vaguely resembles a small crouched person when viewed from the camera.

The object must remain clearly an object when inspected closely.

Do NOT create a mannequin.
Do NOT create a human silhouette.
Do NOT create a statue.
Do NOT include a person wearing the object.

The surrounding environment should contain realistic snow variation, footprints from earlier activity, exposed rocks, small plants, uneven snow depth, and shadows.

The camera should be conducting a normal search sweep and should not deliberately frame the ambiguous object.

Use realistic camera imperfections, including subtle stabilization movement, vibration, exposure changes, motion blur, sensor noise, compression, and atmospheric haze.

No cinematic staging.
No dramatic lighting.
No text.
No HUD.
No bounding boxes.
No watermark.
```

---

# Snow — Clear Negative

```text
Create approximately 10 seconds of photorealistic operational field footage over an empty snowy search area.

NO PERSON.
NO HUMAN-LIKE OBJECT.
NO CLOTHING.
NO BACKPACK.
NO MANNEQUIN.
NO STATUE.
NO INTENTIONAL HUMAN-SHAPED DECOY.

The camera is conducting a routine search sweep over ordinary snow-covered terrain.

Show realistic variation in the snow:
wind patterns, compacted snow, footprints from previous non-visible activity, partially exposed rocks, small vegetation, uneven terrain, and natural shadows.

Do not make the environment pristine or cinematic.

The camera is stabilized but has subtle real-world vibration and micro-movement.

Include realistic sensor noise, compression, motion blur, and exposure adaptation.

The scene must remain continuous and empty for the entire clip.

No person should appear in the distance or at the edge of the frame.

No text.
No HUD.
No bounding boxes.
No watermark.
```

---

# Altitude — Positive

```text
Create approximately 10 seconds of photorealistic operational field footage from a relatively HIGH elevated viewpoint.

The camera looks substantially downward toward the ground, approximately 50–70 degrees downward.

A SINGLE real adult person is visible on the ground.

Because the camera is relatively high, the person must appear SMALL in the image, approximately 0.5–2% of the frame area.

The person should be naturally walking across ordinary terrain and should NOT pose, wave, look toward the camera, or move unnaturally.

The camera is conducting a systematic search sweep.

Do not lock onto the person and do not zoom toward them.

Use realistic ground detail and perspective appropriate for a high elevated viewpoint:
large patches of terrain, vegetation patterns, rocks, trails, shadows, and irregular ground features.

The person should be genuinely visible but difficult to notice at first glance.

Use realistic camera characteristics:
moderate compression, limited fine detail, atmospheric haze, slight motion blur, sensor noise, stabilization artifacts, and realistic exposure.

Do not artificially enlarge or sharpen the person.

The footage must look like an operational search recording, not a cinematic aerial photograph.

No dramatic zoom.
No subject tracking.
No centered person.
No text.
No HUD.
No bounding boxes.
No watermark.
```

---

# Altitude — Hard Negative

```text
Create approximately 10 seconds of photorealistic operational field footage from a relatively HIGH elevated viewpoint.

THERE MUST BE NO HUMAN BEING PRESENT.

The camera looks substantially downward toward ordinary ground terrain.

Include a naturally occurring small ambiguous pattern that could potentially trigger a person detector from this distance.

For example, a dark piece of fabric or abandoned object partially overlapping a rock and casting a small shadow should create a vague human-like pattern when viewed from above.

The object must remain a real object and must NOT form a recognizable human silhouette.

Because the camera is high, the ambiguous object should be small in the image.

The camera is performing a normal search sweep and should not zoom toward or deliberately center the object.

Use realistic elevated perspective, atmospheric haze, compression, limited detail, motion blur, sensor noise, and minor stabilization artifacts.

The environment should contain ordinary terrain variation rather than a perfectly clean landscape.

NO person.
NO mannequin.
NO statue.
NO human body parts.
NO intentionally constructed human dummy.

No cinematic aerial photography.
No dramatic zoom.
No text.
No HUD.
No bounding boxes.
No watermark.
```

---

# Altitude — Clear Negative

```text
Create approximately 10 seconds of photorealistic operational field footage from a relatively HIGH elevated viewpoint.

The camera looks substantially downward toward the ground.

There must be absolutely NO PERSON anywhere in the scene.

There must also be NO clothing, backpacks, mannequins, statues, human-shaped objects, or deliberate detector decoys.

Show ordinary terrain viewed from a high elevated viewpoint:
vegetation patterns, rocks, dirt, trails, shadows, small naturally occurring structures if present, uneven terrain, and atmospheric haze.

The scene should look visually ordinary and somewhat difficult to interpret at first glance because of the distance.

The camera moves slowly and steadily during a practical search sweep.

Use realistic camera imperfections:
compression, limited spatial detail, sensor noise, atmospheric haze, slight motion blur, subtle stabilization movement, and exposure adaptation.

Do not create a beautiful cinematic landscape.

No dramatic camera movement.
No zoom.
No artificial sharpening.
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
