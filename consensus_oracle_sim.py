from manim import *
import numpy as np
import random
import math
from manim.utils.color import interpolate_color

class OracleConsensusScene(Scene):
    def construct(self):
        random.seed(42)
        np.random.seed(42)

        n_nodes = 6
        n_bits = 6

        # Threshold for 2/3 of broadcasters (excluding self)
        threshold = math.ceil(2 * (n_nodes - 1) / 3)

        # Assign a unique base color per node and precompute lighter query colors
        node_colors = [BLUE, PURPLE, ORANGE, PINK, TEAL, GOLD]
        query_colors = [interpolate_color(c, WHITE, 0.5) for c in node_colors]

        # Create the oracle (a random array of 0s and 1s)
        oracle_bits = np.random.randint(0, 2, size=n_bits)

        # Create Oracle visualization
        bit_boxes = VGroup()
        bit_positions = []
        for idx, bit in enumerate(oracle_bits):
            box = Square(side_length=0.6)
            box.shift(RIGHT * idx * 0.7)
            text = Text(str(bit), font_size=24).move_to(box.get_center())
            bit_boxes.add(VGroup(box, text))
        bit_boxes.move_to(UP * 3)
        oracle_label = Text("Oracle", font_size=30).next_to(bit_boxes, UP)

        # Record oracle cell positions
        for bit_box in bit_boxes:
            bit_positions.append(bit_box.get_center())

        # Create network nodes and memory arrays with smaller radius
        radius = 2.2  # reduced from 2.5 for a more compact layout
        nodes = []
        memory_arrays = []
        for i in range(n_nodes):
            angle = i * TAU / n_nodes
            pos = radius * np.array([np.cos(angle), np.sin(angle), 0])
            base_color = node_colors[i]
            circle = Circle(radius=0.3, color=base_color, fill_color=BLACK, fill_opacity=1).move_to(pos)
            label = Text(str(i), font_size=24, color=base_color).move_to(pos)
            memory = VGroup()
            for j in range(n_bits):
                offset = RIGHT * (j - n_bits/2) * 0.5
                cell_x = (pos + RIGHT * 2.2 + offset) if i in (0, 1, 5) else (pos + LEFT * 1.8 + offset)
                mem_box = Text("[ ]", font_size=18, color=GRAY).move_to(cell_x)
                memory.add(mem_box)
            node_group = VGroup(circle, label, memory)
            nodes.append(node_group)
            memory_arrays.append(memory)

        # Create edges between nodes
        edges = VGroup(*[
            Line(nodes[i][0].get_center(), nodes[j][0].get_center(), color=GRAY)
            for i in range(n_nodes) for j in range(i+1, n_nodes)
        ])

        # Fade in network and oracle simultaneously (slower)
        self.wait(1)
        self.play(
            FadeIn(edges, run_time=1.5),
            *[FadeIn(n, run_time=1.5) for n in nodes],
            FadeIn(oracle_label, run_time=1.5),
            FadeIn(bit_boxes, run_time=1.5)
        )

        # Track self-verifications and receive counts
        verified = [[False] * n_bits for _ in range(n_nodes)]
        receive_counts = [[{0: set(), 1: set()} for _ in range(n_bits)] for _ in range(n_nodes)]

        # Simulation rounds
        for _ in range(n_bits):
            self.wait(0.6)  # slower pacing

            # Query phase: each node picks an unverified bit
            queries = []
            indices = []
            lines = []
            for i in range(n_nodes):
                choices = [b for b in range(n_bits) if not verified[i][b]]
                k = random.choice(choices) if choices else random.randint(0, n_bits-1)
                val = oracle_bits[k]
                queries.append(val)
                indices.append(k)
                # thicker query line in node's lighter color
                line = Line(
                    nodes[i][0].get_center(), bit_positions[k],
                    color=query_colors[i], stroke_width=4
                )
                lines.append(line)
                self.add(line)

            self.wait(1)
            self.play(*[FadeOut(l, run_time=0.8) for l in lines])

            # Self-verification: memory turns white
            self_anims = []
            for i, k in enumerate(indices):
                if not verified[i][k]:
                    old = memory_arrays[i][k]
                    new = Text(f"[{queries[i]}]", font_size=18, color=WHITE).move_to(old.get_center())
                    self_anims.append(Transform(old, new))
                    verified[i][k] = True
            if self_anims:
                self.play(*self_anims, run_time=0.7)

            # Broadcast phase: dots still red/green for values
            dots = []
            for i in range(n_nodes):
                for j in range(n_nodes):
                    if i != j:
                        b_idx = indices[i]
                        val = queries[i]
                        dot = Dot(color=GREEN if val == 1 else RED).scale(0.6)
                        dot.move_to(nodes[i][0].get_center())
                        dots.append((dot, i, j, b_idx, val))
                        self.add(dot)

            self.play(*[
                d.animate.move_to(nodes[j][0].get_center()) for d,_,j,_,_ in dots
            ], run_time=1.5)  # slower broadcast
            for d, *_ in dots:
                self.remove(d)

            # Receive update: 2/3 threshold or blend (slower)
            recv_anims = []
            for _, sender, receiver, b_idx, val in dots:
                if not verified[receiver][b_idx]:
                    receive_counts[receiver][b_idx][val].add(sender)
                    count = len(receive_counts[receiver][b_idx][val])
                    old = memory_arrays[receiver][b_idx]
                    if count >= threshold:
                        verified[receiver][b_idx] = True
                        new = Text(f"[{val}]", font_size=18, color=WHITE).move_to(old.get_center())
                    else:
                        alpha = count / (n_nodes - 1)
                        clr = interpolate_color(RED, GREEN, alpha)
                        new = Text(f"[{val}]", font_size=18, color=clr).move_to(old.get_center())
                    recv_anims.append(Transform(old, new))
            if recv_anims:
                self.play(*recv_anims, run_time=0.7)

        self.wait(2)
