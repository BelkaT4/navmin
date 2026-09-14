#include "navmin_control.h"
#include "navmin_motion.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ASSERT_TRUE(expr) do { \
    if (!(expr)) { \
        fprintf(stderr, "%s:%d: assertion failed: %s\n", __FILE__, __LINE__, #expr); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

#define ASSERT_EQ_I32(expected, actual) do { \
    int32_t expected_value_ = (int32_t)(expected); \
    int32_t actual_value_ = (int32_t)(actual); \
    if (expected_value_ != actual_value_) { \
        fprintf(stderr, "%s:%d: expected %ld, got %ld\n", \
                __FILE__, __LINE__, \
                (long)expected_value_, (long)actual_value_); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

#define ASSERT_EQ_I64(expected, actual) do { \
    int64_t expected_value_ = (int64_t)(expected); \
    int64_t actual_value_ = (int64_t)(actual); \
    if (expected_value_ != actual_value_) { \
        fprintf(stderr, "%s:%d: expected %lld, got %lld\n", \
                __FILE__, __LINE__, \
                (long long)expected_value_, (long long)actual_value_); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

#define ASSERT_EQ_U32(expected, actual) do { \
    uint32_t expected_value_ = (uint32_t)(expected); \
    uint32_t actual_value_ = (uint32_t)(actual); \
    if (expected_value_ != actual_value_) { \
        fprintf(stderr, "%s:%d: expected %lu, got %lu\n", \
                __FILE__, __LINE__, \
                (unsigned long)expected_value_, (unsigned long)actual_value_); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

typedef struct {
    uint32_t positive_x;
    uint32_t negative_x;
    uint32_t positive_y;
    uint32_t negative_y;
    uint32_t calls_this_tick_x;
    uint32_t calls_this_tick_y;
    uint32_t max_calls_per_tick_x;
    uint32_t max_calls_per_tick_y;
} step_sink_t;

static void emit_step(
    void *context,
    navmin_motion_axis_t axis,
    navmin_step_direction_t direction
)
{
    step_sink_t *sink = context;

    if (axis == NAVMIN_MOTION_AXIS_X) {
        ++sink->calls_this_tick_x;
        if (sink->calls_this_tick_x > sink->max_calls_per_tick_x) {
            sink->max_calls_per_tick_x = sink->calls_this_tick_x;
        }
        if (direction == NAVMIN_STEP_DIRECTION_POSITIVE) {
            ++sink->positive_x;
        } else {
            ++sink->negative_x;
        }
    } else {
        ++sink->calls_this_tick_y;
        if (sink->calls_this_tick_y > sink->max_calls_per_tick_y) {
            sink->max_calls_per_tick_y = sink->calls_this_tick_y;
        }
        if (direction == NAVMIN_STEP_DIRECTION_POSITIVE) {
            ++sink->positive_y;
        } else {
            ++sink->negative_y;
        }
    }
}

static navmin_motion_limits_t make_limits(
    uint32_t max_x,
    uint32_t max_y,
    uint32_t acceleration_x,
    uint32_t acceleration_y
)
{
    navmin_motion_limits_t limits;

    limits.max_speed_x_steps_s = max_x;
    limits.max_speed_y_steps_s = max_y;
    limits.acceleration_x_steps_s2 = acceleration_x;
    limits.acceleration_y_steps_s2 = acceleration_y;
    return limits;
}

static navmin_motion_t make_motion(
    step_sink_t *sink,
    navmin_motion_limits_t limits
)
{
    navmin_motion_t motion;
    navmin_motion_step_sink_t step_sink;

    memset(sink, 0, sizeof(*sink));
    step_sink.context = sink;
    step_sink.emit_step = emit_step;
    navmin_motion_init(&motion, step_sink);
    ASSERT_TRUE(navmin_motion_set_limits(&motion, limits));
    return motion;
}

static void tick(navmin_motion_t *motion, step_sink_t *sink)
{
    sink->calls_this_tick_x = 0U;
    sink->calls_this_tick_y = 0U;
    navmin_motion_control_tick(motion);
}

static void run_ticks(navmin_motion_t *motion, step_sink_t *sink, uint32_t count)
{
    uint32_t i;

    for (i = 0U; i < count; ++i) {
        tick(motion, sink);
    }
}

static void run_until_targets(navmin_motion_t *motion, step_sink_t *sink)
{
    uint32_t guard = 0U;

    while (motion->x.current_velocity_q != motion->x.target_velocity_q ||
           motion->y.current_velocity_q != motion->y.target_velocity_q) {
        tick(motion, sink);
        ++guard;
        ASSERT_TRUE(guard < 1000000U);
    }
}

static void test_init_is_stationary(void)
{
    navmin_motion_t motion;
    navmin_motion_step_sink_t empty_sink = {0};
    uint32_t i;

    navmin_motion_init(&motion, empty_sink);
    ASSERT_EQ_I32(0, motion.x.current_velocity_q);
    ASSERT_EQ_I32(0, motion.y.current_velocity_q);
    ASSERT_EQ_I32(0, motion.x.target_velocity_q);
    ASSERT_EQ_I32(0, motion.y.target_velocity_q);
    ASSERT_EQ_I64(0, motion.x.commanded_position_steps);
    ASSERT_EQ_I64(0, motion.y.commanded_position_steps);
    ASSERT_TRUE(!motion.limits_valid);

    for (i = 0U; i < NAVMIN_CONTROL_RATE_HZ; ++i) {
        navmin_motion_control_tick(&motion);
    }
    ASSERT_EQ_I64(0, motion.x.commanded_position_steps);
    ASSERT_EQ_I64(0, motion.y.commanded_position_steps);
}

static void test_positive_constant_velocity_exact_count(void)
{
    step_sink_t sink;
    navmin_motion_t motion = make_motion(
        &sink, make_limits(10000U, 10000U, 100000U, 100000U));
    uint32_t before;

    navmin_motion_set_velocity_target(&motion, 1000, 0);
    run_until_targets(&motion, &sink);
    before = sink.positive_x;
    run_ticks(&motion, &sink, NAVMIN_CONTROL_RATE_HZ);

    ASSERT_EQ_U32(1000U, sink.positive_x - before);
    ASSERT_EQ_U32(0U, sink.negative_x);
    ASSERT_EQ_I64((int64_t)sink.positive_x, motion.x.commanded_position_steps);
}

static void test_negative_constant_velocity_exact_count_and_direction(void)
{
    step_sink_t sink;
    navmin_motion_t motion = make_motion(
        &sink, make_limits(10000U, 10000U, 100000U, 100000U));
    uint32_t before;

    navmin_motion_set_velocity_target(&motion, -1000, 0);
    run_until_targets(&motion, &sink);
    before = sink.negative_x;
    run_ticks(&motion, &sink, NAVMIN_CONTROL_RATE_HZ);

    ASSERT_EQ_U32(1000U, sink.negative_x - before);
    ASSERT_EQ_U32(0U, sink.positive_x);
    ASSERT_EQ_I64(-(int64_t)sink.negative_x, motion.x.commanded_position_steps);
}

static void test_max_velocity_is_exact_and_at_most_one_step_per_axis_tick(void)
{
    step_sink_t sink;
    navmin_motion_t motion = make_motion(
        &sink, make_limits(10000U, 10000U, 100000U, 100000U));
    uint32_t before;

    navmin_motion_set_velocity_target(&motion, 10000, 0);
    run_until_targets(&motion, &sink);
    before = sink.positive_x;
    run_ticks(&motion, &sink, NAVMIN_CONTROL_RATE_HZ);

    ASSERT_EQ_U32(10000U, sink.positive_x - before);
    ASSERT_TRUE(sink.max_calls_per_tick_x <= 1U);
}

static void test_axes_schedule_independently(void)
{
    step_sink_t sink;
    navmin_motion_t motion = make_motion(
        &sink, make_limits(10000U, 10000U, 100000U, 100000U));
    uint32_t x_before;
    uint32_t y_before;

    navmin_motion_set_velocity_target(&motion, 1000, -2500);
    run_until_targets(&motion, &sink);
    x_before = sink.positive_x;
    y_before = sink.negative_y;
    run_ticks(&motion, &sink, NAVMIN_CONTROL_RATE_HZ);

    ASSERT_EQ_U32(1000U, sink.positive_x - x_before);
    ASSERT_EQ_U32(2500U, sink.negative_y - y_before);
    ASSERT_TRUE(sink.max_calls_per_tick_x <= 1U);
    ASSERT_TRUE(sink.max_calls_per_tick_y <= 1U);
}

static void test_acceleration_limits_each_tick(void)
{
    step_sink_t sink;
    navmin_motion_t motion = make_motion(
        &sink, make_limits(10000U, 10000U, 50000U, 100000U));
    int32_t previous = 0;
    uint32_t i;

    navmin_motion_set_velocity_target(&motion, 1000, 0);
    for (i = 0U; i < 100U; ++i) {
        tick(&motion, &sink);
        ASSERT_TRUE(motion.x.current_velocity_q - previous <= 50000);
        ASSERT_TRUE(motion.x.current_velocity_q >= previous);
        previous = motion.x.current_velocity_q;
    }
}

static void test_deceleration_to_zero_is_limited_and_clears_phase(void)
{
    step_sink_t sink;
    navmin_motion_t motion = make_motion(
        &sink, make_limits(10000U, 10000U, 100000U, 100000U));
    int32_t previous;

    navmin_motion_set_velocity_target(&motion, 1000, 0);
    run_until_targets(&motion, &sink);
    navmin_motion_set_velocity_target(&motion, 0, 0);
    previous = motion.x.current_velocity_q;

    while (motion.x.current_velocity_q != 0) {
        tick(&motion, &sink);
        ASSERT_TRUE(previous - motion.x.current_velocity_q <= 100000);
        ASSERT_TRUE(motion.x.current_velocity_q >= 0);
        previous = motion.x.current_velocity_q;
    }
    ASSERT_EQ_I32(0, motion.x.step_phase_q);
}

static void test_reversal_passes_through_zero_before_opposite_acceleration(void)
{
    step_sink_t sink;
    navmin_motion_t motion = make_motion(
        &sink, make_limits(10000U, 10000U, 100000U, 100000U));
    bool saw_zero = false;

    navmin_motion_set_velocity_target(&motion, 1000, 0);
    run_until_targets(&motion, &sink);
    navmin_motion_set_velocity_target(&motion, -1000, 0);

    while (motion.x.current_velocity_q >= 0) {
        tick(&motion, &sink);
        if (motion.x.current_velocity_q == 0) {
            saw_zero = true;
            ASSERT_EQ_I32(0, motion.x.step_phase_q);
            break;
        }
    }
    ASSERT_TRUE(saw_zero);
    tick(&motion, &sink);
    ASSERT_TRUE(motion.x.current_velocity_q < 0);
}

static void test_reversal_discards_unused_acceleration_budget_at_zero(void)
{
    step_sink_t sink;
    navmin_motion_t motion = make_motion(
        &sink, make_limits(10000U, 10000U, 50000U, 50000U));
    navmin_motion_limits_t faster = make_limits(10000U, 10000U, 100000U, 50000U);

    navmin_motion_set_velocity_target(&motion, 1000, 0);
    tick(&motion, &sink);
    ASSERT_EQ_I32(50000, motion.x.current_velocity_q);

    ASSERT_TRUE(navmin_motion_set_limits(&motion, faster));
    navmin_motion_set_velocity_target(&motion, -1000, 0);
    tick(&motion, &sink);
    ASSERT_EQ_I32(0, motion.x.current_velocity_q);
    tick(&motion, &sink);
    ASSERT_EQ_I32(-100000, motion.x.current_velocity_q);
}

static void test_target_clamps_and_speed_increase_does_not_restore_hidden_request(void)
{
    step_sink_t sink;
    navmin_motion_t motion = make_motion(
        &sink, make_limits(1000U, 1000U, 100000U, 100000U));

    navmin_motion_set_velocity_target(&motion, 2000, INT32_MIN);
    ASSERT_EQ_I32(1000 * (int32_t)NAVMIN_VELOCITY_SCALE, motion.x.target_velocity_q);
    ASSERT_EQ_I32(-1000 * (int32_t)NAVMIN_VELOCITY_SCALE, motion.y.target_velocity_q);

    ASSERT_TRUE(navmin_motion_set_limits(
        &motion, make_limits(3000U, 3000U, 100000U, 100000U)));
    ASSERT_EQ_I32(1000 * (int32_t)NAVMIN_VELOCITY_SCALE, motion.x.target_velocity_q);
    ASSERT_EQ_I32(-1000 * (int32_t)NAVMIN_VELOCITY_SCALE, motion.y.target_velocity_q);

    navmin_motion_set_velocity_target(&motion, 2000, 0);
    ASSERT_EQ_I32(2000 * (int32_t)NAVMIN_VELOCITY_SCALE, motion.x.target_velocity_q);
}

static void test_runtime_speed_decrease_clamps_target_but_not_current(void)
{
    step_sink_t sink;
    navmin_motion_t motion = make_motion(
        &sink, make_limits(3000U, 3000U, 100000U, 100000U));
    int32_t current_before;

    navmin_motion_set_velocity_target(&motion, 2000, 0);
    run_until_targets(&motion, &sink);
    current_before = motion.x.current_velocity_q;

    ASSERT_TRUE(navmin_motion_set_limits(
        &motion, make_limits(1000U, 3000U, 100000U, 100000U)));
    ASSERT_EQ_I32(1000 * (int32_t)NAVMIN_VELOCITY_SCALE, motion.x.target_velocity_q);
    ASSERT_EQ_I32(current_before, motion.x.current_velocity_q);

    tick(&motion, &sink);
    ASSERT_EQ_I32(current_before - 100000, motion.x.current_velocity_q);
}

static void test_acceleration_change_applies_on_next_tick(void)
{
    step_sink_t sink;
    navmin_motion_t motion = make_motion(
        &sink, make_limits(10000U, 10000U, 20000U, 20000U));

    navmin_motion_set_velocity_target(&motion, 1000, 0);
    tick(&motion, &sink);
    ASSERT_EQ_I32(20000, motion.x.current_velocity_q);

    ASSERT_TRUE(navmin_motion_set_limits(
        &motion, make_limits(10000U, 10000U, 100000U, 20000U)));
    tick(&motion, &sink);
    ASSERT_EQ_I32(120000, motion.x.current_velocity_q);
}

static void test_hard_stop_clears_numeric_motion_and_phase_but_preserves_position(void)
{
    step_sink_t sink;
    navmin_motion_t motion = make_motion(
        &sink, make_limits(10000U, 10000U, 100000U, 100000U));
    int64_t position_before;
    uint32_t calls_before;

    navmin_motion_set_velocity_target(&motion, 1000, 0);
    run_until_targets(&motion, &sink);
    run_ticks(&motion, &sink, 7U);
    ASSERT_TRUE(motion.x.step_phase_q != 0);
    position_before = motion.x.commanded_position_steps;
    calls_before = sink.positive_x + sink.negative_x;

    navmin_motion_hard_stop(&motion);
    ASSERT_EQ_I32(0, motion.x.current_velocity_q);
    ASSERT_EQ_I32(0, motion.x.target_velocity_q);
    ASSERT_EQ_I32(0, motion.x.step_phase_q);
    ASSERT_EQ_I64(position_before, motion.x.commanded_position_steps);

    run_ticks(&motion, &sink, NAVMIN_CONTROL_RATE_HZ);
    ASSERT_EQ_U32(calls_before, sink.positive_x + sink.negative_x);
    ASSERT_EQ_I64(position_before, motion.x.commanded_position_steps);
}

static void test_zero_velocity_long_run_emits_no_steps(void)
{
    step_sink_t sink;
    navmin_motion_t motion = make_motion(
        &sink, make_limits(10000U, 10000U, 100000U, 100000U));

    run_ticks(&motion, &sink, NAVMIN_CONTROL_RATE_HZ * 3U);
    ASSERT_EQ_U32(0U, sink.positive_x + sink.negative_x +
                       sink.positive_y + sink.negative_y);
    ASSERT_EQ_I64(0, motion.x.commanded_position_steps);
    ASSERT_EQ_I64(0, motion.y.commanded_position_steps);
}

static void test_position_changes_only_when_step_callback_is_emitted(void)
{
    navmin_motion_t motion;
    navmin_motion_step_sink_t empty_sink = {0};
    navmin_motion_limits_t limits = make_limits(10000U, 10000U, 100000U, 100000U);

    navmin_motion_init(&motion, empty_sink);
    ASSERT_TRUE(navmin_motion_set_limits(&motion, limits));
    navmin_motion_set_velocity_target(&motion, 10000, -10000);
    {
        uint32_t i;
        for (i = 0U; i < NAVMIN_CONTROL_RATE_HZ * 2U; ++i) {
            navmin_motion_control_tick(&motion);
        }
    }
    ASSERT_EQ_I64(0, motion.x.commanded_position_steps);
    ASSERT_EQ_I64(0, motion.y.commanded_position_steps);
}

static void test_stop_restart_does_not_reuse_old_fractional_phase(void)
{
    step_sink_t sink;
    navmin_motion_t motion = make_motion(
        &sink, make_limits(10000U, 10000U, 100000U, 100000U));

    navmin_motion_set_velocity_target(&motion, 1000, 0);
    run_until_targets(&motion, &sink);
    run_ticks(&motion, &sink, 7U);
    ASSERT_TRUE(motion.x.step_phase_q != 0);

    navmin_motion_hard_stop(&motion);
    ASSERT_EQ_I32(0, motion.x.step_phase_q);
    navmin_motion_set_velocity_target(&motion, 1, 0);
    tick(&motion, &sink);
    ASSERT_TRUE(motion.x.step_phase_q > 0);
    ASSERT_TRUE(motion.x.step_phase_q < NAVMIN_STEP_PHASE_THRESHOLD);
}

static void test_extreme_allowed_limits_and_requested_int32_values_are_safe(void)
{
    step_sink_t sink;
    navmin_motion_t motion = make_motion(
        &sink,
        make_limits(
            NAVMIN_MAX_SUPPORTED_STEP_RATE,
            NAVMIN_MAX_SUPPORTED_STEP_RATE,
            NAVMIN_MAX_SUPPORTED_ACCELERATION,
            NAVMIN_MAX_SUPPORTED_ACCELERATION));

    navmin_motion_set_velocity_target(&motion, INT32_MAX, INT32_MIN);
    ASSERT_EQ_I32(200000000, motion.x.target_velocity_q);
    ASSERT_EQ_I32(-200000000, motion.y.target_velocity_q);
    run_ticks(&motion, &sink, 100000U);
    ASSERT_TRUE(sink.max_calls_per_tick_x <= 1U);
    ASSERT_TRUE(sink.max_calls_per_tick_y <= 1U);
    ASSERT_EQ_I64(
        (int64_t)sink.positive_x - (int64_t)sink.negative_x,
        motion.x.commanded_position_steps);
    ASSERT_EQ_I64(
        (int64_t)sink.positive_y - (int64_t)sink.negative_y,
        motion.y.commanded_position_steps);
}

static void test_invalid_limits_do_not_replace_previous_limits(void)
{
    step_sink_t sink;
    navmin_motion_t motion = make_motion(
        &sink, make_limits(1000U, 2000U, 3000U, 4000U));
    navmin_motion_limits_t original = motion.limits;

    ASSERT_TRUE(!navmin_motion_set_limits(
        &motion, make_limits(0U, 2000U, 3000U, 4000U)));
    ASSERT_TRUE(memcmp(&motion.limits, &original, sizeof(original)) == 0);
    ASSERT_TRUE(!navmin_motion_set_limits(
        &motion,
        make_limits(
            NAVMIN_MAX_SUPPORTED_STEP_RATE + 1U,
            2000U,
            3000U,
            4000U)));
    ASSERT_TRUE(memcmp(&motion.limits, &original, sizeof(original)) == 0);
}

int main(void)
{
    test_init_is_stationary();
    test_positive_constant_velocity_exact_count();
    test_negative_constant_velocity_exact_count_and_direction();
    test_max_velocity_is_exact_and_at_most_one_step_per_axis_tick();
    test_axes_schedule_independently();
    test_acceleration_limits_each_tick();
    test_deceleration_to_zero_is_limited_and_clears_phase();
    test_reversal_passes_through_zero_before_opposite_acceleration();
    test_reversal_discards_unused_acceleration_budget_at_zero();
    test_target_clamps_and_speed_increase_does_not_restore_hidden_request();
    test_runtime_speed_decrease_clamps_target_but_not_current();
    test_acceleration_change_applies_on_next_tick();
    test_hard_stop_clears_numeric_motion_and_phase_but_preserves_position();
    test_zero_velocity_long_run_emits_no_steps();
    test_position_changes_only_when_step_callback_is_emitted();
    test_stop_restart_does_not_reuse_old_fractional_phase();
    test_extreme_allowed_limits_and_requested_int32_values_are_safe();
    test_invalid_limits_do_not_replace_previous_limits();

    puts("navmin firmware motion host tests: PASS (18 cases)");
    return EXIT_SUCCESS;
}
