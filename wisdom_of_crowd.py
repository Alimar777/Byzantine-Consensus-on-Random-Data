from manim import *
import random

# Global animation speed multiplier
animation_speed = 2.0

def darken(color, amount=0.5):
    return interpolate_color(color, BLACK, amount)

class WisdomOfTheCrowd(Scene):
    def construct(self):
        jar = RoundedRectangle(corner_radius=0.5, height=4, width=2.5)
        jar.set_stroke(WHITE, width=4)
        jar.set_fill(BLUE_E, opacity=0.1)
        jar.to_edge(LEFT, buff=1)
        self.play(FadeIn(jar), run_time=1.0 * animation_speed)

        jellybeans = VGroup()
        colors = [RED, GREEN, BLUE, YELLOW, ORANGE, PURPLE, PINK, TEAL, GOLD]
        stroke_colors = [darken(c, amount=0.5) for c in colors]

        for _ in range(500):
            color_choice = random.choice(list(zip(colors, stroke_colors)))
            bean = Ellipse(width=0.18, height=0.09, color=color_choice[0], fill_opacity=1, stroke_color=color_choice[1], stroke_width=1)
            bean.rotate(random.uniform(-PI/4, PI/4))
            x = random.uniform(-1.1, 1.1)
            y = random.uniform(-1.8, 1.8)
            bean.move_to(jar.get_center() + x * RIGHT + y * UP)
            jellybeans.add(bean)

        self.play(LaggedStart(*[FadeIn(bean, scale=0.5) for bean in jellybeans], lag_ratio=0.002), run_time=2.0 * animation_speed)

        number_line = NumberLine(x_range=[0, 1000, 100], length=8)
        number_line.shift(DOWN * 2.5)
        self.play(Create(number_line), run_time=1.0 * animation_speed)

        true_count = 500
        honest_guesses = [int(random.gauss(mu=true_count, sigma=100)) for _ in range(14)]
        honest_guesses = [max(0, min(1000, g)) for g in honest_guesses]
        byzantine_guesses = [random.choice([0, 1000]) for _ in range(6)]

        guesses = honest_guesses + byzantine_guesses
        random.shuffle(guesses)
        is_byzantine = [g in [0, 1000] for g in guesses]

        guess_column_label = Text("Guesses", font_size=28).to_edge(RIGHT, buff=0.8).shift(UP * 2.5)
        guess_title = Underline(guess_column_label)
        guess_texts = VGroup()
        for i, guess in enumerate(guesses):
            color = RED if is_byzantine[i] else YELLOW
            text = Text(f"{guess}", font_size=20, color=color)
            text.next_to(guess_title, DOWN, aligned_edge=LEFT).shift(DOWN * (i * 0.28))
            guess_texts.add(text)

        self.play(FadeIn(guess_column_label, guess_title), run_time=1.0 * animation_speed)

        guess_dots = VGroup()

        avg_tracker = ValueTracker(0)
        median_tracker = ValueTracker(0)
        mom_tracker = ValueTracker(0)

        true_label = Text(f"True: {true_count}", font_size=24, color=GREEN)
        true_label.next_to(jar, RIGHT, buff=0.4).align_to(jar, UP)

        avg_text_pos = true_label.get_left() + DOWN * 0.6
        median_text_pos = avg_text_pos + DOWN * 0.6
        mom_text_pos = median_text_pos + DOWN * 0.6

        avg_text = always_redraw(lambda: Text(f"Avg: {avg_tracker.get_value():.1f}", font_size=24, color=BLUE).move_to(avg_text_pos, aligned_edge=LEFT))
        median_text = always_redraw(lambda: Text(f"Median: {median_tracker.get_value():.1f}", font_size=24, color=ORANGE).move_to(median_text_pos, aligned_edge=LEFT))
        mom_text = always_redraw(lambda: Text(f"Med of Means: {mom_tracker.get_value():.1f}", font_size=24, color=PURPLE).move_to(mom_text_pos, aligned_edge=LEFT))

        self.play(FadeIn(true_label, avg_text, median_text, mom_text), run_time=1.0 * animation_speed)

        avg_marker = always_redraw(lambda: Triangle(color=BLUE, fill_opacity=1).scale(0.2).rotate(PI).next_to(number_line.n2p(avg_tracker.get_value()), UP, buff=0.1))
        median_marker = always_redraw(lambda: Triangle(color=ORANGE, fill_opacity=1).scale(0.2).next_to(number_line.n2p(median_tracker.get_value()), DOWN, buff=0.1))
        mom_marker = always_redraw(lambda: Triangle(color=PURPLE, fill_opacity=1).scale(0.2).next_to(number_line.n2p(mom_tracker.get_value()), DOWN, buff=0.1))

        true_marker = Triangle(color=GREEN, fill_opacity=1).scale(0.2).rotate(PI)
        true_marker.next_to(number_line.n2p(true_count), UP, buff=0.1)
        self.play(FadeIn(true_marker, avg_marker, median_marker, mom_marker), run_time=1.0 * animation_speed)

        for i, (guess, byz) in enumerate(zip(guesses, is_byzantine)):
            color = RED if byz else YELLOW
            text = guess_texts[i]
            dot = Dot(color=color).move_to(number_line.n2p(guess))

            self.play(FadeIn(text), GrowFromCenter(dot), run_time=0.5 * animation_speed)
            guess_dots.add(dot)

            current_guesses = guesses[:i+1]
            avg = sum(current_guesses) / len(current_guesses)
            med = sorted(current_guesses)[len(current_guesses) // 2]
            group_chunks = [sorted(current_guesses[j:j+5]) for j in range(0, len(current_guesses), 5)]
            moms = [g[len(g)//2] for g in group_chunks if g]
            mom = sum(moms) / len(moms)

            self.play(
                avg_tracker.animate.set_value(avg),
                median_tracker.animate.set_value(med),
                mom_tracker.animate.set_value(mom),
                run_time=0.4 * animation_speed
            )

        close_text = Text("Median of Means Resists Byzantine guesses!", font_size=32, color=YELLOW).to_edge(UP, buff=0.5)
        self.wait(1 * animation_speed)
        self.play(Write(close_text), run_time=1.0 * animation_speed)
        self.wait(2 * animation_speed)
        self.play(FadeOut(guess_dots), run_time=1.0 * animation_speed)
