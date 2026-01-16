from manim import *
import random
import numpy as np

USE_FIXED_SEED = True
SEED = 42

class MultiplicativeWeightsAnimation(Scene):
    def construct(self):
        if USE_FIXED_SEED:
            np.random.seed(SEED)
            random.seed(SEED)

        n = 9
        rounds = 6
        c = 1.0
        eta = 0.4

        byzantine_indices = set(random.sample(range(n), n // 3))
        spacing = 1.2

        bars, weight_labels = [], []
        bar_width = 0.3
        initial_height = 2.0
        weights = [1.0 for _ in range(n)]
        base_y = 0.3

        x_offsets = [(i - (n - 1) / 2) * spacing for i in range(n)]

        for i, x in enumerate(x_offsets):
            bar = Rectangle(width=bar_width, height=initial_height, color=GREEN, fill_opacity=0.7).move_to(RIGHT * x + UP * base_y, aligned_edge=DOWN)
            label = MathTex(f"{weights[i]:.2f}", font_size=30).next_to(bar, DOWN, buff=0.08)
            bars.append(bar)
            weight_labels.append(label)

        self.play(AnimationGroup(*[GrowFromEdge(bar, edge=DOWN) for bar in bars], *[FadeIn(label, shift=UP) for label in weight_labels], lag_ratio=0.1))

        dots, x_labels = [], []
        for i, x in enumerate(x_offsets):
            color = RED if i in byzantine_indices else BLUE
            dot = Dot(color=color).next_to(weight_labels[i], DOWN, buff=0.1)
            label = MathTex(f"X_{{{i+1}}}", font_size=30).next_to(dot, DOWN, buff=0.08)
            dots.append(dot)
            x_labels.append(label)

        self.play(AnimationGroup(*[FadeIn(dot, shift=UP) for dot in dots], *[FadeIn(label, shift=UP) for label in x_labels], lag_ratio=0.1))

        line_range = 6
        num_line = NumberLine(x_range=[-line_range, line_range, 1], length=10, include_tip=False).to_edge(DOWN)
        self.play(FadeIn(num_line))

        # Initialize persistent sample markers
        sample_markers = []
        for i in range(n):
            color = RED if i in byzantine_indices else BLUE
            marker = Dot(color=color, radius=0.05).move_to(num_line.n2p(0))
            sample_markers.append(marker)
            self.add(marker)

        mu_text = None
        mu_marker = None

        honest_index = next(i for i in range(n) if i not in byzantine_indices)
        byzantine_index = next(iter(byzantine_indices))
        tracked_indices = sorted([honest_index, byzantine_index])

        loss_eqs, weight_eqs = [], []
        eq_y = num_line.get_top()[1] + 1.5
        equation_spacing = 4
        x_offset = [-equation_spacing, equation_spacing]

        for j, idx in enumerate(tracked_indices):
            initial_loss = MathTex(rf"\min_{{{idx+1}}}\left(1, \frac{{0}}{{{c}}}\right) = 0.00", font_size=32)
            initial_weight = MathTex(rf"w_{{{idx+1}}} \leftarrow 1.00", font_size=32)
            initial_loss.move_to([x_offset[j], eq_y, 0])
            initial_weight.next_to(initial_loss, DOWN, buff=0.25)
            loss_eqs.append(initial_loss)
            weight_eqs.append(initial_weight)
            self.play(FadeIn(initial_loss), FadeIn(initial_weight))

        for round_num in range(rounds):
            round_text = Text(f"Round {round_num+1}", font_size=36).to_corner(UL)
            self.play(FadeIn(round_text))
            self.wait(0.5)

            x_samples = []
            for i in range(n):
                if i in byzantine_indices:
                    x_i = np.random.normal(5 if random.random() < 0.5 else -5, 5.0)
                else:
                    x_i = np.random.normal(0, 0.3)
                x_samples.append(x_i)

            mu_hat = np.median(x_samples)
            new_mu_text = MathTex(rf"\hat{{\mu}} = {mu_hat:.2f}", font_size=32).next_to(num_line, UP)
            new_mu_marker = Dot(color=YELLOW).move_to(num_line.n2p(mu_hat))

            if mu_text and mu_marker:
                self.play(Transform(mu_text, new_mu_text), Transform(mu_marker, new_mu_marker))
            else:
                mu_text = new_mu_text
                mu_marker = new_mu_marker
                self.play(FadeIn(mu_text), FadeIn(mu_marker))

            clipped_samples = np.clip(x_samples, -line_range, line_range)
            animations = [
                sample_markers[i].animate.move_to(num_line.n2p(clipped_samples[i]))
                for i in range(n)
            ]
            self.play(*animations, run_time=1.5)

            losses = []
            for i, x_i in enumerate(x_samples):
                loss = min(1.0, abs(x_i - mu_hat) / c)
                losses.append(loss)

            for i in range(n):
                weights[i] *= (1 - eta * losses[i])

            animations, updates = [], []
            for i, x in enumerate(x_offsets):
                new_height = initial_height * weights[i]
                new_bar = Rectangle(width=bar_width, height=new_height, color=GREEN, fill_opacity=0.7)
                new_bar.move_to(RIGHT * x + UP * base_y, aligned_edge=DOWN)
                animations.append(Transform(bars[i], new_bar))
                weight_eq = MathTex(f"{weights[i]:.2f}", font_size=30)
                weight_eq.next_to(new_bar, DOWN, buff=0.05)
                updates.append(Transform(weight_labels[i], weight_eq))

            self.play(AnimationGroup(*animations, *updates, lag_ratio=0.1), run_time=2)

            for j, idx in enumerate(tracked_indices):
                x_i = x_samples[idx]
                loss = losses[idx]
                w_prev = weights[idx] / (1 - eta * loss)
                w_new = weights[idx]

                new_loss_eq = MathTex(
                    rf"\min_{{{idx+1}}} \left(1, \frac{{|{x_i:.2f} - {mu_hat:.2f}|}}{{{c}}} \right) = {loss:.2f}", font_size=32
                ).move_to(loss_eqs[j].get_center())
                new_weight_eq = MathTex(
                    rf"w_{{{idx+1}}} \leftarrow {w_prev:.2f} \cdot (1 - {eta} \cdot {loss:.2f}) = {w_new:.2f}", font_size=32
                ).move_to(weight_eqs[j].get_center())

                self.play(Transform(loss_eqs[j], new_loss_eq), Transform(weight_eqs[j], new_weight_eq))

            normalize_text = Text("Normalize", font_size=36).to_corner(UL)
            self.play(Transform(round_text, normalize_text))
            self.wait(0.5)

            max_weight = max(weights)
            if max_weight > 0:
                weights = [w / max_weight for w in weights]

            animations, updates = [], []
            for i, x in enumerate(x_offsets):
                new_height = initial_height * weights[i]
                new_bar = Rectangle(width=bar_width, height=new_height, color=GREEN, fill_opacity=0.7)
                new_bar.move_to(RIGHT * x + UP * base_y, aligned_edge=DOWN)
                animations.append(Transform(bars[i], new_bar))
                weight_eq = MathTex(f"{weights[i]:.2f}", font_size=30)
                weight_eq.next_to(new_bar, DOWN, buff=0.05)
                updates.append(Transform(weight_labels[i], weight_eq))

            self.play(AnimationGroup(*animations, *updates, lag_ratio=0.1), run_time=2)
            self.play(FadeOut(round_text))

        self.wait(2)
