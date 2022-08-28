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
    fn test_forbidden_check4() {
        #[rustfmt::skip]
        let mut board = BoardMatrix::from_row_slice(&[
            0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ]);

        assert_eq!(board.is_forbidden((3, 5)), true);

        board[(4, 6)] = 1;

        assert_eq!(board.is_forbidden((3, 5)), false);

        board[(5, 7)] = 1;

        assert_eq!(board.is_forbidden((3, 5)), true);

        board[(3, 4)] = 1;
        assert_eq!(board.is_forbidden((3, 5)), false);
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
        assert_eq!(state.get_white_win_moves().len(), 1);
        assert_eq!(state.get_white_win_moves()[0], (0, 0));

        let state = board.scan_continuous_pieces((1, 3), Color::Black);
        assert_eq!(state.get_black_win_moves().len(), 2);
        assert_eq!(state.get_black_win_moves()[0], (1, 0));
        assert_eq!(state.get_black_win_moves()[1], (1, 5));

        let state = board.scan_continuous_pieces((5, 6), Color::Black);
        assert_eq!(state.get_black_win_moves().len(), 1);
        assert_eq!(state.get_black_win_moves()[0], (4, 5));

        assert_eq!(board.is_forbidden((4, 1)), true);
        let state = board.scan_continuous_pieces((4, 1), Color::Black);
        assert_eq!(state.get_black_win_moves().len(), 3);
        assert_eq!(state.get_black_win_moves()[0], (2, 1));
        assert_eq!(state.get_black_win_moves()[1], (6, 1));
        assert_eq!(state.get_black_win_moves()[2], (3, 2));
    }
}
