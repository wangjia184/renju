

#[cfg(test)]
mod game_test {
    use match_generator::game::*;
    
    #[test] 
    fn test_check_position_state() {

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
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);

        let state = board.check_black_in_position( (0, 4) );
        assert_eq!(state.get_open_four_count(), 1);
        assert_eq!(state.get_live_three_count(), 0);
        
        let state = board.check_black_in_position( (0, 0) );
        assert_eq!(state.get_open_four_count(), 1);
        assert_eq!(state.get_live_three_count(), 0);

        let state = board.check_black_in_position( (2, 8) );
        assert_eq!(state.get_open_four_count(), 1);
        assert_eq!(state.get_live_three_count(), 0);

        let state = board.check_black_in_position( (2, 5) );
        assert_eq!(state.get_open_four_count(), 1);
        assert_eq!(state.get_live_three_count(), 0);

        let state = board.check_black_in_position( (2, 3) );
        assert_eq!(state.get_open_four_count(), 1);
        assert_eq!(state.get_live_three_count(), 0);

        let state = board.check_black_in_position( (4, 9) );
        assert_eq!(state.get_open_four_count(), 1);
        assert_eq!(state.get_live_three_count(), 0);

        let state = board.check_black_in_position( (4, 10) );
        assert_eq!(state.get_open_four_count(), 1);
        assert_eq!(state.get_live_three_count(), 0);


        let state = board.check_black_in_position( (6, 1) );
        assert_eq!(state.get_open_four_count(), 0);
        assert_eq!(state.get_live_three_count(), 1);

        let state = board.check_black_in_position( (6, 5) );
        assert_eq!(state.get_open_four_count(), 0);
        assert_eq!(state.get_live_three_count(), 1);

        let state = board.check_black_in_position( (9, 9) );
        assert_eq!(state.get_open_four_count(), 2);
        assert_eq!(state.get_live_three_count(), 0);

        let state = board.check_black_in_position( (12, 6) );
        assert_eq!(state.get_open_four_count(), 0);
        assert_eq!(state.get_live_three_count(), 0);

        let state = board.check_black_in_position( (12, 1) );
        assert_eq!(state.get_open_four_count(), 0);
        assert_eq!(state.get_live_three_count(), 1);

        let state = board.check_black_in_position( (12, 11) );
        assert_eq!(state.get_open_four_count(), 0);
        assert_eq!(state.get_live_three_count(), 1);
    }



    #[test] 
    fn test_forbidden_check2() {

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
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,]);

        let state = board.check_black_in_position( (7, 7) );
        assert_eq!(state.get_open_four_count(), 2);
        assert_eq!(state.get_live_three_count(), 0);
        assert_eq!(board.is_forbidden((7, 7)), true);

        let state = board.check_black_in_position( (8, 4) );
        assert_eq!(state.get_open_four_count(), 2);
        assert_eq!(state.get_live_three_count(), 0);
        assert_eq!(board.is_forbidden((8, 4)), true);

        let state = board.check_black_in_position( (5, 10) );
        assert_eq!(state.get_open_four_count(), 2);
        assert_eq!(state.get_live_three_count(), 0);
        assert_eq!(board.is_forbidden((5, 10)), true);
    }


    #[test] 
    fn test_forbidden_check3() {

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
            0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0,]);

        assert_eq!(board.is_forbidden((9, 7)), true);

        board[(9, 9)] = 1;

        assert_eq!(board.is_forbidden((9, 7)), false);

        assert_eq!(board.is_forbidden((2, 10)), true);

        board[(5, 10)] = 2;

        assert_eq!(board.is_forbidden((2, 10)), false);

        assert_eq!(board.is_forbidden((14, 6)), true);
    }






    #[test]
    fn test_pattern()
    {
    BoardMatrix::generate_opening_patterns().values().enumerate().for_each( |(idx, x)| {
        println!("{}", idx + 1);
        x.print();
    });
    
    }



    #[test]
    fn test_for_each_piece()
    {
    let mut m = BoardMatrix::zeros();
    m[(0,0)] = 1;
    m[(0,1)] = 1;
    m[(0,2)] = 1;
    m[(0,3)] = 2;
    m[(0,4)] = 2;
    m[(0,5)] = 1;
    m[(0,6)] = 2;
    m[(0,7)] = 2;
    m[(0,8)] = 1;
    
    m.for_each_piece( | row, col, value | {
        println!("{}:{} = {}", row, col, value);
    })
    }


    #[test]
    fn test_is_over()
    {
    let mut m = BoardMatrix::zeros();
    m[(6,6)] = 1;
    m[(7,7)] = 1;
    m[(8,8)] = 1;
    m[(9,9)] = 1;
    m[(10,10)] = 2;
    m[(11,11)] = 1;
    m[(12,12)] = 1;
    m[(13,13)] = 1;
    m[(14,14)] = 1;
    
    assert_eq!( m.is_over(), false);
    m[(9,9)] = 2;
    m[(10,10)] = 1;
    assert_eq!( m.is_over(), true);

    let mut m = BoardMatrix::zeros();
    m[(0,0)] = 2;
    m[(0,1)] = 2;
    m[(0,2)] = 2;
    m[(0,4)] = 2;
    assert_eq!( m.is_over(), false);
    m[(0,3)] = 2;
    assert_eq!( m.is_over(), true);


    let mut m = BoardMatrix::zeros();
    m[(0,14)] = 1;
    m[(0,13)] = 1;
    m[(0,12)] = 1;
    m[(0,11)] = 1;
    assert_eq!( m.is_over(), false);
    m[(0,10)] = 1;
    assert_eq!( m.is_over(), true);


    let mut m = BoardMatrix::zeros();
    m[(0,14)] = 1;
    m[(1,14)] = 1;
    m[(2,14)] = 1;
    m[(3,14)] = 1;
    m[(4,14)] = 2;
    assert_eq!( m.is_over(), false);
    m[(4,14)] = 1;
    assert_eq!( m.is_over(), true);

    let m = BoardMatrix::from_base81_string("0000000000000i0003r00R200p904QS08?017N01J2020001000000000");
    m.print();
    assert_eq!( m.is_over(), false);
    }


    #[test]
    fn test_all_appearances(){
    let mut m = BoardMatrix::zeros();
    m[(0,0)] = 1;
    m[(0,1)] = 2;
    m[(1,0)] = 1;

    m.get_all_appearances((1,1)).iter().for_each( |(board, answer)| {
        board.print();
        println!("{:?}", answer);
    });
    }
}

