from manim import *
import numpy as np
import random
import math
from manim.utils.color import interpolate_color

class OracleConsensusScene(Scene):
    def construct(self):
        random.seed(42)
        np.random.seed(42)

        randomize_oracle = True  # <<<<<< SWITCH TO RANDOMIZE ORACLE OUTPUTS

        n_nodes = 6
        n_bits = 6
        beta = 1/3
        byzantine_count = math.floor(beta * n_nodes)
        byzantine_nodes = random.sample(list(range(n_nodes)), byzantine_count)

        committee_factor = 2.5
        base_size = max(3, int(committee_factor * math.log(n_nodes)))
        min_majority_size = 2 * byzantine_count + 1
        committee_size = min(n_nodes, max(base_size, min_majority_size))
        threshold_committee = math.ceil(2 * committee_size / 3)

        node_colors = [BLUE, PURPLE, ORANGE, PINK, TEAL, GOLD]
        query_colors = [interpolate_color(c, WHITE, 0.5) for c in node_colors]

        oracle_bits = np.random.randint(0, 2, size=n_bits)
        bit_boxes = VGroup()
        for idx, bit in enumerate(oracle_bits):
            box = Square(side_length=0.6, fill_color=BLACK, fill_opacity=1)
            if randomize_oracle:
                text = MathTex(f"P_{{{idx+1}}}", font_size=24, color=WHITE)
            else:
                text = Text(str(bit), font_size=24, color=WHITE)
            bit_boxes.add(VGroup(box, text))
        bit_boxes.arrange(RIGHT, buff=0.7).move_to(UP * 3)
        bit_positions = [g[0].get_center() for g in bit_boxes]
        for g in bit_boxes:
            g[1].move_to(g[0].get_center())
        oracle_label = Text(
            "Randomized Oracle" if randomize_oracle else "Perfect Oracle",
            font_size=30,
            color=YELLOW
        ).next_to(bit_boxes, UP)

        radius = 1.8
        nodes = []
        memory_arrays = []
        for i in range(n_nodes):
            angle = i * TAU / n_nodes
            pos = radius * np.array([math.cos(angle), math.sin(angle), 0])
            back_circ = Circle(0.3, fill_color=BLACK, fill_opacity=1, stroke_width=0).move_to(pos)
            circ = Circle(0.3, stroke_color=node_colors[i], fill_color=BLACK, fill_opacity=1).move_to(pos)
            lbl = Text(str(i), font_size=24, color=WHITE).move_to(pos)
            mem = VGroup()
            for j in range(n_bits):
                offset = RIGHT * (j - n_bits/2) * 0.5
                cell = Text("[ ]", font_size=18, color=GRAY).move_to(
                    pos + (RIGHT * 2.2 + offset if i in (0,1,5) else LEFT * 1.8 + offset)
                )
                mem.add(cell)
            nodes.append(VGroup(back_circ, circ, lbl, mem))
            memory_arrays.append(mem)

        edges = {}
        network_lines = []
        for i in range(n_nodes):
            for j in range(i+1, n_nodes):
                line = Line(nodes[i][1].get_center(), nodes[j][1].get_center(), color=GRAY)
                edges[(i,j)] = edges[(j,i)] = line
                network_lines.append(line)

        self.wait(1)
        self.play(
            FadeIn(*network_lines, run_time=1.5),
            *[FadeIn(n, run_time=1.5) for n in nodes],
            FadeIn(bit_boxes, run_time=1.5),
            FadeIn(oracle_label, run_time=1.5)
        )
        self.capture_screenshot(name="scene_start")

        verified = [[False]*n_bits for _ in range(n_nodes)]
        receive_counts = [[{0:set(),1:set()} for _ in range(n_bits)] for _ in range(n_nodes)]
        blacklisted_edges = set()

        for k in range(n_bits):
            committee = random.sample(range(n_nodes), committee_size)

            highlight_anims = []
            for i in range(n_nodes):
                _, circ, _, _ = nodes[i]
                if i in committee:
                    highlight_anims.append(circ.animate.set_fill(node_colors[i], opacity=0.5))
                else:
                    highlight_anims.append(circ.animate.set_fill(BLACK, opacity=1))
            self.play(*highlight_anims, run_time=0.5)
            self.capture_screenshot(name=f"query_step_{k}_before")

            query_lines = []
            vals = {}
            for i in committee:
                if randomize_oracle:
                    sampled_bit = random.choice([0,1])
                else:
                    sampled_bit = oracle_bits[k]
                true = sampled_bit
                v = 1-true if i in byzantine_nodes else true
                vals[i] = v
                ln = Line(nodes[i][1].get_center(), bit_positions[k], color=query_colors[i], stroke_width=4)
                query_lines.append(ln)
                self.add(ln)
            self.add_foreground_mobject(bit_boxes)
            self.wait(1)
            self.play(*[FadeOut(ln, run_time=0.5) for ln in query_lines])

            reset_anims = [nodes[i][1].animate.set_fill(BLACK, opacity=1) for i in range(n_nodes)]
            self.play(*reset_anims, run_time=0.5)

            anims = []
            for i in committee:
                if not verified[i][k]:
                    old = memory_arrays[i][k]
                    new = Text(f"[{vals[i]}]", font_size=18, color=WHITE).move_to(old.get_center())
                    anims.append(Transform(old, new))
                    verified[i][k] = True
            if anims:
                self.play(*anims, run_time=0.5)
            self.capture_screenshot(name=f"query_step_{k}_after")

            visual_dots = []
            for i in committee:
                for j in range(n_nodes):
                    if j == i:
                        continue
                    receive_counts[j][k][vals[i]].add(i)
                    if (min(i, j), max(i, j)) not in blacklisted_edges:
                        dot = Dot(color=GREEN if vals[i]==1 else RED).scale(0.8)
                        dot.move_to(nodes[i][1].get_center())
                        visual_dots.append((dot, i, j))
                        self.add(dot)
            self.play(*[d.animate.move_to(nodes[j][1].get_center()) for d,_,j in visual_dots], run_time=1)
            for d,_,_ in visual_dots:
                self.remove(d)

            fades = []
            global_S0 = set()
            global_S1 = set()

            for sender in vals:
                if vals[sender] == 0:
                    global_S0.add(sender)
                else:
                    global_S1.add(sender)

            if global_S0 and global_S1:
                minority = global_S1 if len(global_S0) > len(global_S1) else global_S0
            else:
                minority = set()

            for liar in minority:
                for r in range(n_nodes):
                    if liar == r:
                        continue
                    key = (min(liar, r), max(liar, r))
                    if key not in blacklisted_edges:
                        fades.append(FadeOut(edges[(liar, r)], run_time=0.5))
                        blacklisted_edges.add(key)

            if fades:
                self.play(*fades)

            anims = []
            for r in range(n_nodes):
                S0 = receive_counts[r][k][0]
                S1 = receive_counts[r][k][1]
                total = len(S0) + len(S1)
                if total < threshold_committee:
                    continue
                if S0 and S1:
                    consensus = oracle_bits[k]
                else:
                    consensus = 0 if len(S0) > len(S1) else 1
                display_val = 1-consensus if r in byzantine_nodes else consensus
                cell = memory_arrays[r][k]
                anims.append(
                    Transform(cell,
                              Text(f"[{display_val}]", font_size=18, color=WHITE)
                              .move_to(cell.get_center()))
                )
                verified[r][k] = True
            if anims:
                self.play(*anims, run_time=0.5)
            self.wait(0.5)

        reveal = Text(f"Byzantine Nodes: {byzantine_nodes}", font_size=24).to_edge(DOWN)
        self.play(FadeIn(reveal, run_time=1))
        color_anims = []
        for i in byzantine_nodes:
            _, circ, lbl, _ = nodes[i]
            color_anims.append(circ.animate.set_fill(RED, opacity=1))
            color_anims.append(lbl.animate.set_color(WHITE))
        self.play(*color_anims, run_time=1)
        for i in byzantine_nodes:
            _, _, lbl, _ = nodes[i]
            self.bring_to_front(lbl)
        self.wait(4)
        self.capture_screenshot(name="scene_end")
