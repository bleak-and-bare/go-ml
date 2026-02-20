module.exports = {
    plugins: {
        "postcss-preset-mantine": {},
        "postcss-simple-vars": {
            variables: {
                "mantine-breakpoint-xs": "360px",
                "mantine-breakpoint-sm": "480px",
                "mantine-breakpoint-md": "640px",
                "mantine-breakpoint-lg": "768px",
                "mantine-breakpoint-xl": "1024px",
            },
        },
    },
};
