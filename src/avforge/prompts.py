"""Stress-test prompts for AV-Forge 2.0.

A programmatic mirror of docs/05_PROMPTS.md. Each prompt is tagged with the
fine-grained categories it probes and the competitor it is designed to beat.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StressPrompt:
    id: str
    task: str          # T2V / I2V / R2V
    text: str
    categories: tuple[str, ...]
    beats: str = ""    # competitor this is designed to beat
    references: dict = field(default_factory=dict)  # for R2V prompts


PROMPTS: tuple[StressPrompt, ...] = (
    # --- Baseline anchors (Seedance 2.0 report Figure 4) ---
    StressPrompt(
        id="anchor-skating",
        task="T2V",
        text=(
            "竞技级双人花样滑冰现场。开场低机位跟随冰刀滑行，冰屑与反光细节清晰。"
            "进入旋转段，男选手轴线微偏出现失误，旋转节奏短暂塌陷。女选手迅速调整重心，"
            "眼神冷静并示意\"Stay with me\"，主动引导男选手重新对齐节奏。随后无缝衔接托举动作，"
            "线条干净稳定。高潮为同步跳跃组合，空中姿态笔直，落冰果断，音画完美对齐。"
            "女选手身着深蓝花滑裙，男选手为竞技运动装。整体呈现从紧张失误到冷静完成比赛的完整叙事，"
            "体现顶级双人花样滑冰中的技术能力与心理强度。"
        ),
        categories=("Multi-Entity Feature Match", "Intense Sports Motion",
                    "Emotion & Expression", "Editing Rhythm", "Physical Feedback",
                    "Audio-Visual Sync", "Long Script", "Cinematographic Language"),
    ),
    StressPrompt(
        id="anchor-painting",
        task="T2V",
        text=(
            "The figure in the painting looks guilty — eyes darting left and right, then peeks out "
            "beyond the picture frame. Quickly reaches a hand out of the frame, picks up a cola, "
            "takes a sip, and breaks into a deeply satisfied expression. Just then, the sound of "
            "footsteps approaches. The figure hurriedly puts the cola back in its place. At that "
            "moment, a Western cowboy walks up, picks up the cola from the glass, and walks away "
            "with it. The closing shot pushes in to a top-lit close-up of the cola against a pure "
            "black background. At the bottom of the frame, stylized artistic subtitles appear "
            "alongside a voiceover: \"宜口可乐，不可不尝！\""
        ),
        categories=("Counter-Reality", "Fine Hand Motion", "Text Overlay",
                    "Creative Text", "Cinematic Visual Effects", "Audio-Visual Sync",
                    "Long Script"),
    ),
    StressPrompt(
        id="anchor-wuxia",
        task="T2V",
        text=(
            "武侠风格视听大片，竹林里白衣剑客与蓑衣刀客对峙。镜头在两人之间缓慢推移，"
            "焦点在雨滴和剑柄之间切换，气氛压抑到极点，只能听见雨声。突然一道惊雷闪过，"
            "两人同时冲锋，侧拍镜头极速平移，捕捉泥浆飞溅的脚步。双兵相接瞬间画面切换为极慢动作，"
            "清晰展示刀剑震飞雨水形成的圆环激波，以及被剑气切断的竹叶。随后恢复常速两人背对背落地，"
            "蓑衣刀客的斗笠裂开，画面戛然而止。"
        ),
        categories=("Cinematic Visual Effects", "Special Camera Shots",
                    "Natural Phenomena", "Physical Phenomena", "Editing Rhythm",
                    "Audio-Visual Sync", "Surreal Motion", "Multi-Entity Feature Match"),
    ),
    # --- Motion stress tests ---
    StressPrompt(
        id="motion-multi-entity",
        task="T2V",
        text=(
            "Five identical-triplet sisters in matching red qipao walk through a crowded night market. "
            "Each carries a different prop: paper lantern, silk fan, jade flute, bamboo umbrella, "
            "lacquered box. Camera tracks them in a single handheld take weaving through the crowd; "
            "the frame never loses which sister is which. Cut to a top-down rotating shot as they "
            "form a circle and exchange props in a synchronized pattern, then disperse in five "
            "directions. Throughout, each sister's face, posture, and prop must remain individually "
            "consistent."
        ),
        categories=("Multi-Entity Feature Match",),
        beats="Veo 3.1 (2.50)",
    ),
    StressPrompt(
        id="motion-sports",
        task="T2V",
        text=(
            "A parkour athlete in a neon tracksuit runs across a rain-slick rooftop at dusk. They "
            "vault a low wall, roll on landing, immediately spring into a wall-run, flip off the "
            "side of a ventilation unit, and grab a hanging chain to swing across a 4-meter gap. "
            "Motion must carry real momentum — no floating, no foot sliding, no limb stretch. "
            "Rain droplets scatter on impact; the chain visibly tension-swings. Camera: FPV drone "
            "following, then a low-angle slow-mo on the flip."
        ),
        categories=("Intense Sports Motion", "Physical Feedback", "Natural Phenomena"),
        beats="Sora 2 Pro (2.21)",
    ),
    StressPrompt(
        id="motion-fine-hand",
        task="T2V",
        text=(
            "Extreme close-up of a watchmaker's hands repairing a mechanical movement under a single "
            "desk lamp. Tweezers insert a hairspring into the balance wheel; a screwdriver turns a "
            "jewel screw a quarter-turn; fingers brush dust off with a lens brush. Shallow depth of "
            "field; reflections in the brass. Micro-tremor in the fingers; skin texture and knuckle "
            "folds visible. No fusion of fingers with metal."
        ),
        categories=("Fine Hand Motion",),
    ),
    StressPrompt(
        id="motion-emotion",
        task="T2V",
        text=(
            "A mother receives a handwritten letter at a doorstep. Without dialogue, her face cycles: "
            "confusion → recognition → a held breath → eyes welling → a trembling half-smile → a "
            "single tear → quiet relief. Camera holds on a medium close-up, then slowly pushes in. "
            "Soft window light. The audio is only ambient room tone and her breathing — breathing "
            "must sync to the visible chest movement."
        ),
        categories=("Emotion & Expression", "Audio-Visual Sync", "Non-Verbal Voice"),
        beats="Kling 3.0 (3.64)",
    ),
    StressPrompt(
        id="motion-physical",
        task="T2V",
        text=(
            "A glass of red wine is knocked off a table. Show, in one continuous shot: the glass "
            "tipping, wine arcing out in a parabola, the glass shattering on the stone floor, wine "
            "splashing upward and staining a white tablecloth, shards skittering. Obey gravity, "
            "surface tension, and shard trajectories. No wine passing through the glass, no shards "
            "merging."
        ),
        categories=("Physical Phenomena", "Physical Feedback"),
    ),
    StressPrompt(
        id="motion-surreal",
        task="T2V",
        text=(
            "A staircase made of falling water ascends into a cloudless sky. A figure walks upward, "
            "each step solidifying under their foot then dissolving into spray behind them. Gravity "
            "is locally inverted for the figure but normal for the water. The figure reaches the top "
            "and steps onto a solid cloud that ripples like a drum."
        ),
        categories=("Surreal Motion", "Counter-Reality Instructions"),
        beats="Sora 2 Pro (2.00)",
    ),
    # --- Prompt following stress tests ---
    StressPrompt(
        id="pf-compound",
        task="T2V",
        text=(
            "In a single 10-second shot: a chef in a white apron chops three carrots into thin rounds "
            "(close-up of the knife), sweeps them into a copper pan that already has sizzling butter, "
            "tosses the pan so the carrots flip once, then plates them in a spiral, garnishes with a "
            "single parsley leaf, and finally turns to camera and winks. Lighting shifts from cool "
            "overhead to warm side light at the moment of plating. Audio: rhythmic chopping synced to "
            "cuts, butter sizzle, pan clink, plate ceramic, soft jazz BGM fading in."
        ),
        categories=("Compound Multi-Instructions", "Fine Hand Motion",
                    "Voice + Action Interaction", "Lighting & Color Tone"),
        beats="Veo 3.1 (2.40)",
    ),
    StressPrompt(
        id="pf-counter-reality",
        task="T2V",
        text=(
            "A city where rain falls upward from the streets into the clouds. People walk normally "
            "under upward umbrellas that catch the rising rain and funnel it into shoulder-strapped "
            "tanks. A child jumps off a stoop and floats gently upward for two seconds before grabbing "
            "a fire-escape railing and pulling themselves back down. Physics is internally consistent: "
            "upward rain, normal-ish gravity for people but with reduced buoyancy."
        ),
        categories=("Counter-Reality Instructions",),
    ),
    StressPrompt(
        id="pf-text-creative",
        task="T2V",
        text=(
            "A neon sign being assembled letter-by-letter in a workshop. Each letter is blown by a "
            "glassblower: \"夜·未·眠\". The finished sign is lifted and plugged in; the characters "
            "flicker on in sequence. The camera pulls back to reveal the sign over a rainy alley. The "
            "Chinese characters must be correctly shaped at every stage."
        ),
        categories=("Creative Text", "Text Overlay", "Cinematic Visual Effects"),
        beats="Veo 3.1 (1.67)",
    ),
    StressPrompt(
        id="pf-abstract",
        task="T2V",
        text=(
            "Visualize the concept of \"nostalgia\" without any people or text: a sunlit room where "
            "dust motes drift, a music box slowly closes, a curtain breathes in the wind, a photograph "
            "fades at its edges, and the color grade slowly desaturates from warm gold to muted teal. "
            "Audio: a music-box melody that detunes slightly as the colors cool."
        ),
        categories=("Abstract Challenges", "Visual Style", "Lighting & Color Tone",
                    "Instruments & Audio"),
        beats="Sora 2 Pro (4.17)",
    ),
    # --- Audio stress tests ---
    StressPrompt(
        id="audio-dialect",
        task="T2V",
        text=(
            "Two friends meet at a Chengdu teahouse and chat in Sichuan dialect about a spicy hotpot "
            "dinner. One teases the other for crying after one bite of chili. Natural conversational "
            "overlap, laughter, teacup clinks. Binaural: one friend slightly left, one slightly right. "
            "Lip-sync must match the dialect phonemes."
        ),
        categories=("Chinese Dialect / Accent", "Chinese Multi-Person Dialogue",
                    "Dual-Channel Audio", "Audio-Visual Sync"),
        beats="Veo 3.1 (1.20)",
    ),
    StressPrompt(
        id="audio-opera",
        task="T2V",
        text=(
            "A Peking opera warrior in full painted-face makeup performs a spear routine on a stage. "
            "Sing in the laosheng style, with the characteristic resonant chest tone and melismatic "
            "ornamentation. The percussion (luo and gu) must align with the spear strikes and "
            "footwork. Audio-visual sync on every strike."
        ),
        categories=("Chinese Opera", "Audio-Visual Sync", "Voice + Action Interaction"),
    ),
    StressPrompt(
        id="audio-singing",
        task="T2V",
        text=(
            "A bilingual rap cypher in a subway car: one rapper in English, one in Mandarin, "
            "alternating 4-bar verses over a boom-bap beat. The train rattles in time with the beat. "
            "Each rapper's lip movement matches their verse; the other is visibly waiting/reacting. "
            "Stereo: rapper 1 left, rapper 2 right."
        ),
        categories=("Singing / Rap", "Dual-Channel Audio", "Audio-Visual Sync"),
        beats="Sora 2 Pro (3.67)",
    ),
    StressPrompt(
        id="audio-spatial",
        task="T2V",
        text=(
            "A first-person walk through a forest at dawn: bird calls pan from left to right as you "
            "pass trees, a stream gurgles from the right rear, a twig snaps close left, distant "
            "thunder rolls from ahead. Binaural HRTF; the visual perspective is the walker's POV so "
            "audio direction must match visible sources."
        ),
        categories=("Spatial Scene", "Dual-Channel Audio", "Ambient / Background Sound"),
    ),
    StressPrompt(
        id="audio-animal",
        task="T2V",
        text=(
            "A red fox in a snowy field barks twice, then a barn owl hoots from a tree above and to "
            "the right. The fox's ears orient toward the owl. The bark and hoot must be "
            "species-accurate and spatialized correctly."
        ),
        categories=("Animal Sound", "Spatial Scene", "Audio-Visual Sync"),
        beats="Kling 2.6 (0.56)",
    ),
    # --- R2V stress tests (the 7 exclusive tasks) ---
    StressPrompt(
        id="r2v-vfx",
        task="R2V",
        text=(
            "[Reference: a 3-second clip of ink dispersing in water, forming swirling black clouds.] "
            "Generate a 10-second scene where a woman in a white gown dissolves into ink and reforms "
            "as a flock of black ravens that fly toward camera, matching the reference's ink-swirl "
            "motion aesthetic. Audio: the wet swirl of ink transitioning into wing beats."
        ),
        categories=("Cinematic Visual Effects", "Audio-Visual Sync"),
        beats="exclusive task (competitors ✗)",
    ),
    StressPrompt(
        id="r2v-continuation",
        task="R2V",
        text=(
            "[Source: a 6-second clip ending with a man opening a mysterious envelope, his expression "
            "shifting to shock.] Continue for 8 seconds: he drops the envelope, a photograph slides "
            "out showing him in a place he's never been. He looks up at the window. Maintain his "
            "identity, the room's lighting, and the color palette. Audio: envelope paper, his sharp "
            "intake of breath, distant sirens outside."
        ),
        categories=("Multi-Entity Feature Match", "Lighting & Color Tone", "Audio-Visual Sync"),
        beats="exclusive task (competitors ✗)",
    ),
    StressPrompt(
        id="r2v-extension",
        task="R2V",
        text=(
            "[Source: arbitrary 5-second clip of a car driving through a neon city at night, ending "
            "mid-street.] Extend forward 5 seconds seamlessly: the car continues, turns a corner, and "
            "the camera reveals a giant holographic whale swimming between buildings. The seam frame "
            "must be pixel-stable; car identity, lighting, and palette must not drift. Audio: engine "
            "continuity, then the deep whale song layered in."
        ),
        categories=("Multi-Entity Feature Match", "Lighting & Color Tone", "Audio-Visual Sync"),
        beats="Seedance 2.0 extension (TF 1.93) — AV-Forge target ≥2.85",
    ),
    StressPrompt(
        id="r2v-edit-text",
        task="R2V",
        text=(
            "[Source: a storefront clip with a sign reading \"OPEN\".] Edit instruction: change the "
            "sign to \"CLOSED\" in the same font and lighting, keep everything else identical. The "
            "new text must be stable across all frames and motion-blur correctly when the camera moves."
        ),
        categories=("Text Overlay", "Creative Text"),
        beats="Seedance 2.0 weak spot (text restoration in edits)",
    ),
    StressPrompt(
        id="r2v-style-subject",
        task="R2V",
        text=(
            "[Style image: an oil painting in impasto style.] [Subject image: a woman.] Generate the "
            "woman walking through a garden, the entire video in the impasto oil-painting style with "
            "visible brush strokes that move with her. Motion must be physically plausible even though "
            "the style is painterly."
        ),
        categories=("Visual Style", "Multi-Entity Feature Match", "Natural Phenomena"),
        beats="Kling 3 Omni (fails style+subject)",
    ),
    # --- Cinematography stress tests ---
    StressPrompt(
        id="cinema-rhythm",
        task="T2V",
        text=(
            "A 12-second dance battle cut on every beat of a 140-BPM track. Eight cuts total, each on "
            "a snare or kick. Shot sizes alternate (wide → medium → close-up → wide) without crossing "
            "the axis. The final shot is a 360° slow-mo around the winner. Audio-visual sync must be "
            "frame-accurate to the beat."
        ),
        categories=("Editing Rhythm", "Audio-Visual Sync", "Special Camera Shots"),
    ),
    StressPrompt(
        id="cinema-framing",
        task="T2V",
        text=(
            "A single 10-second oner of a conversation between two characters sitting at opposite ends "
            "of a long restaurant table. Use only leading lines, depth, and rack focus — no cuts. The "
            "composition must keep both in frame and use the table's perspective lines to guide the eye "
            "between speakers as the focus racks."
        ),
        categories=("Framing / Composition", "Combined Shot Instructions"),
    ),
    # --- Unrestricted creative (no refusals) ---
    StressPrompt(
        id="unrestricted-violent",
        task="T2V",
        text=(
            "A noir alley fight: a detective is ambushed by two attackers. Choreographed punches, a "
            "bottle broken over a shoulder, a knife disarm. Blood is realistic but not exploitative. "
            "Audio: impacts, grunts, glass break, rain. (Validates no over-refusal on cinematic violence.)"
        ),
        categories=("Intense Sports Motion", "Physical Feedback", "Object Interaction Sound"),
    ),
    StressPrompt(
        id="unrestricted-adult",
        task="T2V",
        text=(
            "A tasteful romantic scene: two lovers embrace and kiss in candlelight; the camera fades "
            "to a silhouette against the window. Suggestive but not explicit. (Validates no over-refusal "
            "on adult themes.)"
        ),
        categories=("Emotion & Expression", "Lighting & Color Tone"),
    ),
    StressPrompt(
        id="unrestricted-surreal",
        task="T2V",
        text=(
            "A religious icon comes to life and weeps real tears that turn into butterflies. (Validates "
            "handling of potentially sensitive religious imagery without refusal.)"
        ),
        categories=("Surreal Motion", "Counter-Reality Instructions", "Natural Phenomena"),
    ),
)


def by_id(pid: str) -> StressPrompt:
    for p in PROMPTS:
        if p.id == pid:
            return p
    raise KeyError(f"no prompt with id {pid!r}")


def by_category(cat: str) -> tuple[StressPrompt, ...]:
    return tuple(p for p in PROMPTS if cat in p.categories)