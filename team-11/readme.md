```bash
cd team-11
docker-compose up -d
```

# Dataset summary

[kaggle source](https://www.kaggle.com/datasets/mrdaniilak/russia-real-estate-2021)

The real estate market in Russia is of two types by `object_type`:

- 0 - Secondary real estate market;
- 2 - New building.

By facade type (`building type`):

- 0 - Don't know
- 1 - Other.
- 2 - panel.
- 3 - Monolithic.
- 4 - Brick.
- 5 - blocky.
- 6 - Wooden.

The number of rooms can also be as 1, 2 or more. However, there is a type of apartment that is called a studio apartment. They are labeled by "-1".
