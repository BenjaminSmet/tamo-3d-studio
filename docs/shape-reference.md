# Shape reference photos

Each actor gets two images: a **look** reference (real hair, makeup, beard) for MetaHuman's
grooms and materials later, and a clean **shape** reference that only the head mesh is made
from. TRELLIS.2 turns everything it sees into geometry, so the shape reference should show
only skin and bone.

A brief for the shape reference:

- **Framing:** straight on, head and neck down to the collarbone, square, head centred and level.
- **Expression:** neutral. Mouth closed with the lips gently together, not pursed or pushed
  forward. Eyes open, looking into the camera.
- **No facial hair.**
- **Brows and lashes:** thin and natural, not drawn in or darkened; no mascara, false lashes or
  eyeliner. Dark, high-contrast brows and lashes come out as raised ridges.
- **Hair:** bald, buzzed, or slicked tight to the head. No bun, braids or volume on top: the
  MetaHuman export sizes the head from the top of the mesh, so tall hair makes it too small.
- **No accessories:** no glasses, earrings, piercings, hats or headbands.
- **Makeup:** none or minimal, no strong highlights or contour.
- **Light:** soft and even from the front, no hard shadows or shiny highlights. Plain
  light-grey background.
- **Same person** as the look reference.
- **Resolution:** at least about 1024 px across the head.

A matching side profile under the same rules is worth keeping. It isn't used yet, but it is
the most likely way to fix the shallow back of the head later.
