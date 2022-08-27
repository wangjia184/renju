#[cfg(test)]

mod game_test {
    use renju_game::game::*;

    #[test]

    fn test_check_position_state() {
        #[rustfmt::skip]
        let board = BoardMatrix::from_row_slice(&[
            0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        ]);

        let state = board.scan_continuous_pieces((0, 4), Color::Black);
        assert_eq!(state.get_open_four_count(), 1);
        assert_eq!(state.get_live_three_count(), 0);

        let state = board.scan_continuous_pieces((0, 0), Color::Black);
        assert_eq!(state.get_open_four_count(), 1);
        assert_eq!(state.get_live_three_count(), 0);

        let state = board.scan_continuous_pieces((2, 8), Color::Black);
        assert_eq!(state.get_open_four_count(), 1);
        assert_eq!(state.get_live_three_count(), 0);

        let state = board.scan_continuous_pieces((2, 5), Color::Black);
        assert_eq!(state.get_open_four_count(), 1);
        assert_eq!(state.get_live_three_count(), 0);

        let state = board.scan_continuous_pieces((2, 3), Color::Black);
        assert_eq!(state.get_open_four_count(), 1);
        assert_eq!(state.get_live_three_count(), 0);

        let state = board.scan_continuous_pieces((4, 9), Color::Black);
        assert_eq!(state.get_open_four_count(), 1);
        assert_eq!(state.get_live_three_count(), 0);

        let state = board.scan_continuous_pieces((4, 10), Color::Black);
        assert_eq!(state.get_open_four_count(), 1);
        assert_eq!(state.get_live_three_count(), 0);

        let state = board.scan_continuous_pieces((6, 1), Color::Black);
        assert_eq!(state.get_open_four_count(), 0);
        assert_eq!(state.get_live_three_count(), 1);

        let state = board.scan_continuous_pieces((6, 5), Color::Black);
        assert_eq!(state.get_open_four_count(), 0);
        assert_eq!(state.get_live_three_count(), 1);

        let state = board.scan_continuous_pieces((9, 9), Color::Black);
        assert_eq!(state.get_open_four_count(), 2);
        assert_eq!(state.get_live_three_count(), 0);

        let state = board.scan_continuous_pieces((12, 6), Color::Black);
        assert_eq!(state.get_open_four_count(), 0);
        assert_eq!(state.get_live_three_count(), 0);

        let state = board.scan_continuous_pieces((12, 1), Color::Black);
        assert_eq!(state.get_open_four_count(), 0);
        assert_eq!(state.get_live_three_count(), 1);

        let state = board.scan_continuous_pieces((12, 11), Color::Black);
        assert_eq!(state.get_open_four_count(), 0);
        assert_eq!(state.get_live_three_count(), 1);
    }

    #[test]
    fn test_forbidden_check2() {
        #[rustfmt::skip]
        let board = BoardMatrix::from_row_slice(&[
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
            0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
            0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ]);

        let state = board.scan_continuous_pieces((7, 7), Color::Black);
        assert_eq!(state.get_open_four_count(), 2);
        assert_eq!(state.get_live_three_count(), 0);
        assert_eq!(board.is_forbidden((7, 7)), true);

        let state = board.scan_continuous_pieces((8, 4), Color::Black);
        assert_eq!(state.get_open_four_count(), 2);
        assert_eq!(state.get_live_three_count(), 0);
        assert_eq!(board.is_forbidden((8, 4)), true);

        let state = board.scan_continuous_pieces((5, 10), Color::Black);
        assert_eq!(state.get_open_four_count(), 2);
        assert_eq!(state.get_live_three_count(), 0);
        assert_eq!(board.is_forbidden((5, 10)), true);
    }

    #[test]
    fn test_forbidden_check3() {
        #[rustfmt::skip]
        let mut board = BoardMatrix::from_row_slice(&[
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 2, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0,
        ]);

        assert_eq!(board.is_forbidden((9, 7)), true);

        board[(9, 9)] = 1;

        assert_eq!(board.is_forbidden((9, 7)), false);

        assert_eq!(board.is_forbidden((2, 10)), true);

        board[(5, 10)] = 2;

        assert_eq!(board.is_forbidden((2, 10)), false);

        assert_eq!(board.is_forbidden((14, 6)), true);
    }

    #[test]
    fn test_next_moves() {
        #[rustfmt::skip]
        let board = BoardMatrix::from_row_slice(&[
            0, 2, 0, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ]);

        let state = board.scan_continuous_pieces((0, 2), Color::White);
        assert_eq!(state.get_next_moves().len(), 1);
        assert_eq!(state.get_next_moves()[0], (0, 0));

        let state = board.scan_continuous_pieces((1, 3), Color::Black);
        assert_eq!(state.get_next_moves().len(), 2);
        assert_eq!(state.get_next_moves()[0], (1, 0));
        assert_eq!(state.get_next_moves()[1], (1, 5));

        let state = board.scan_continuous_pieces((5, 6), Color::Black);
        assert_eq!(state.get_next_moves().len(), 1);
        assert_eq!(state.get_next_moves()[0], (4, 5));

        assert_eq!(board.is_forbidden((4, 1)), true);
        let state = board.scan_continuous_pieces((4, 1), Color::Black);
        assert_eq!(state.get_next_moves().len(), 3);
        assert_eq!(state.get_next_moves()[0], (2, 1));
        assert_eq!(state.get_next_moves()[1], (6, 1));
        assert_eq!(state.get_next_moves()[2], (3, 2));
    }

    #[test]
    fn test_is_over() {
        let mut m = BoardMatrix::zeros();
        m[(6, 6)] = 1;
        m[(7, 7)] = 1;
        m[(8, 8)] = 1;
        m[(9, 9)] = 1;
        m[(10, 10)] = 2;
        m[(11, 11)] = 1;
        m[(12, 12)] = 1;
        m[(13, 13)] = 1;
        m[(14, 14)] = 1;

        assert_eq!(m.is_over(), false);
        m[(9, 9)] = 2;
        m[(10, 10)] = 1;
        assert_eq!(m.is_over(), true);

        let mut m = BoardMatrix::zeros();
        m[(0, 0)] = 2;
        m[(0, 1)] = 2;
        m[(0, 2)] = 2;
        m[(0, 4)] = 2;
        assert_eq!(m.is_over(), false);
        m[(0, 3)] = 2;
        assert_eq!(m.is_over(), true);

        let mut m = BoardMatrix::zeros();
        m[(0, 14)] = 1;
        m[(0, 13)] = 1;
        m[(0, 12)] = 1;
        m[(0, 11)] = 1;
        assert_eq!(m.is_over(), false);
        m[(0, 10)] = 1;
        assert_eq!(m.is_over(), true);

        let mut m = BoardMatrix::zeros();
        m[(0, 14)] = 1;
        m[(1, 14)] = 1;
        m[(2, 14)] = 1;
        m[(3, 14)] = 1;
        m[(4, 14)] = 2;
        assert_eq!(m.is_over(), false);
        m[(4, 14)] = 1;
        assert_eq!(m.is_over(), true);

        let m = BoardMatrix::from_base81_string(
            "0000000000000i0003r00R200p904QS08?017N01J2020001000000000",
        );
        m.print();
        assert_eq!(m.is_over(), false);
    }
}
