() => {
    const env = window.__ENV__;
    const engine = window.GUI_ENGINE;
    const events = [];
    window.maiGymAcceptedEvents = events;

    function context(state) {
        const route = engine.select.route(state);
        return {
            route_id: route.params?.id,
            draft: structuredClone(state.draft),
            predictions: structuredClone(state.predictions),
        };
    }

    const reduce = engine.reduce;
    engine.reduce = (state, input) => {
        const before = context(state);
        const next = reduce(state, input);
        const action = env.trajectory().at(-1);
        if (action) events.push({action, context: before});
        return next;
    };

    const step = engine.step;
    engine.step = (state, action) => {
        const before = context(state);
        const result = step(state, action);
        if (result.ok) events.push({action, context: before});
        return result;
    };
}
